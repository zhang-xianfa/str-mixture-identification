import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# ----------------------
# 配置参数
# ----------------------
NUM_GENE_LOCI = 16  # 固定16个基因座
GENE_ORDER = [
    "D8S1179", "D21S11", "D7S820", "CSF1PO",
    "D3S1358", "TH01", "D13S317", "D16S539",
    "D2S1338", "D19S433", "vWA", "TPOX",
    "D18S51", "AMEL", "D5S818", "FGA"
]  # 需与数据文件中的Marker顺序一致
MAX_SIZE = 500  # 根据数据中最大Size值调整
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------
# 1. 真值文件加载（处理字符型AM基因座）
# ----------------------
def load_ground_truth(truth_file):
    truth_df = pd.read_excel(truth_file)
    truth_dict = {}
    for _, row in truth_df.iterrows():
        contrib_id = row['Sample ID']
        alleles = {}
        for i, locus in enumerate(GENE_ORDER):
            alleles_str = str(row[locus]).strip().upper()
            if locus == "AMEL":
                # 解析AM基因座（X/Y）
                if "," in alleles_str:
                    a1, a2 = alleles_str.split(',')
                else:
                    a1, a2 = alleles_str[0], alleles_str[1] if len(alleles_str) == 2 else ('X', 'X')
                alleles[locus] = (a1, a2)
            else:
                # 解析数值型基因座（如12,13）
                a1, a2 = map(float, alleles_str.split(','))
                alleles[locus] = (a1, a2)
        truth_dict[contrib_id] = alleles
    return truth_dict


# ----------------------
# 2. 自定义数据集类（按每16行解析一个样本）
# ----------------------
class STRDataset(Dataset):
    def __init__(self, data_file, truth_file, max_size=MAX_SIZE):
        self.data_df = pd.read_excel(data_file)
        self.truth_dict = load_ground_truth(truth_file)
        self.max_size = max_size
        self.scaler = StandardScaler()

        # 验证数据行数是否为16的倍数
        if len(self.data_df) % NUM_GENE_LOCI != 0:
            raise ValueError("数据行数必须为16的倍数（每个样本16行）")

        # 按每16行分组，每组为一个混合样本
        self.samples = [
            self.data_df.iloc[i:i + NUM_GENE_LOCI]
            for i in range(0, len(self.data_df), NUM_GENE_LOCI)
        ]

        # 检查数据列是否包含必要字段
        required_cols = ['num', 'Marker', 'Size 1', 'Height 1', 'Size 2', 'Height 2']
        if not set(required_cols).issubset(self.data_df.columns):
            raise ValueError("数据文件缺少必要列：num, Marker, Size 1, Height 1, Size 2, Height 2")

        self.locus_matrix = self._preprocess_loci()
        self.labels = self._generate_labels()

    def _preprocess_loci(self):
        """
        强制转换基因座名称为统一格式（去除空格、转大写）后匹配，并生成峰高矩阵
        """
        matrix_list = []
        for sample_group in self.samples:  # 每组16行对应一个混合样本
            sample_matrix = np.zeros((NUM_GENE_LOCI, self.max_size))  # 16基因座×max_size矩阵

            for row_idx, (_, row) in enumerate(sample_group.iterrows()):
                # 强制处理基因座名称：去除所有空格、转大写，确保匹配
                raw_marker = str(row['Marker'])  # 转为字符串避免类型错误
                processed_marker = raw_marker.replace(" ", "").upper()  # 去除空格并转大写
                expected_marker = GENE_ORDER[row_idx].replace(" ", "").upper()  # 统一处理预期名称

                # 宽松匹配：允许名称缩写或格式差异（如D8S1179与D8S1179-）
                if not processed_marker.startswith(expected_marker):
                    raise ValueError(
                        f"基因座匹配失败：\n"
                        f"原始值: {raw_marker}\n"
                        f"处理后: {processed_marker}\n"
                        f"预期: {expected_marker}"
                    )

                # 解析等位基因Size和Height（处理可能的缺失值）
                size1 = row['Size 1']
                height1 = row['Height 1'] if not pd.isna(row['Height 1']) else 0.0
                size2 = row['Size 2']
                height2 = row['Height 2'] if not pd.isna(row['Height 2']) else 0.0

                # 第一个等位基因
                if 1 <= size1 <= self.max_size:
                    col_idx1 = int(size1) - 1
                    sample_matrix[row_idx, col_idx1] += height1  # 累加峰高（支持混合样本）

                # 第二个等位基因
                if 1 <= size2 <= self.max_size:
                    col_idx2 = int(size2) - 1
                    sample_matrix[row_idx, col_idx2] += height2

            # 列标准化：对每个Size位置进行Z-score标准化
            for col in range(self.max_size):
                col_data = sample_matrix[:, col].reshape(-1, 1)
                if np.any(col_data != 0):  # 非全零列才标准化
                    sample_matrix[:, col] = self.scaler.fit_transform(col_data).flatten()

            matrix_list.append(sample_matrix)

        return np.array(matrix_list)

    def _generate_labels(self):
        """从Sample File中解析贡献者ID并生成64维标签（处理AMEL基因座字符型数据）"""
        labels = []
        for sample in self.samples:
            # 从Sample File列获取样本信息（假设每样本组第一行的Sample File有效）
            sample_file = sample.iloc[0]['Sample File']
            if pd.isna(sample_file):
                raise ValueError("Sample File列为空")

            try:
                # 解析贡献者ID：格式为*-ID1_ID2-...，提取第二个-和第三个-之间的部分
                parts = sample_file.split('-')
                if len(parts) < 3:
                    raise ValueError("Sample File格式错误，缺少ID部分")
                ids_part = parts[2]  # 例如："44_45"
                contrib_ids = list(map(int, ids_part.split('_')))[:2]  # 取前两个编号
            except Exception as e:
                raise ValueError(f"贡献者ID解析失败：{sample_file}，错误原因：{e}")

            contrib1, contrib2 = contrib_ids

            # 验证真值文件中是否存在贡献者ID
            if contrib1 not in self.truth_dict or contrib2 not in self.truth_dict:
                raise ValueError(f"贡献者ID不存在于真值文件：{contrib1}/{contrib2}")

            label = []
            for locus in GENE_ORDER:
                # 处理贡献者1的基因型
                alleles1 = self.truth_dict[contrib1][locus]

                # 处理贡献者2的基因型（区分AMEL基因座）
                if locus == "AMEL":  # 文档中基因座名称为"AMEL"
                    alleles2 = self.truth_dict[contrib2][locus]
                    # 将X/Y转换为数值标签（X=0，Y=1）
                    b1 = 0 if alleles2[0] == "X" else 1
                    b2 = 0 if alleles2[1] == "X" else 1
                else:
                    alleles2 = self.truth_dict[contrib2][locus]
                    b1, b2 = alleles2  # 数值型基因座直接使用浮点值

                # 拼接等位基因（a1, a2, b1, b2）
                if locus == "AMEL":
                    # AMEL基因座的alleles1可能为字符型，需转换为数值
                    a1 = 0 if alleles1[0] == "X" else 1
                    a2 = 0 if alleles1[1] == "X" else 1
                    label.extend([a1, a2, b1, b2])
                else:
                    label.extend([alleles1[0], alleles1[1], b1, b2])

            # 确保所有标签为浮点型（AMEL的数值标签自动转换为float）
            labels.append(np.array(label, dtype=np.float32))

        return np.array(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.locus_matrix[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ----------------------
# 3. CNN模型定义
# ----------------------
class STRGenotypeCNN(nn.Module):
    def __init__(self, input_channels=NUM_GENE_LOCI, output_dim=64):
        super(STRGenotypeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (MAX_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # (batch, 128, size//8)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ----------------------
# 4. 训练与评估流程
# ----------------------
def train_model(train_loader, model, criterion, optimizer, epochs=EPOCHS):
    model.to(DEVICE)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        print(f'Epoch {epoch + 1} 平均损失: {total_loss / len(train_loader):.4f}')


def evaluate_model(test_loader, model, truth_dict, gene_order):
    model.eval()
    total_mae = 0
    total_samples = 0
    total_correct = 0
    total_alleles = 0
    predictions = []
    truths = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(data)

            # 转换为numpy数组
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 解析真实值与预测值（按基因座拆分）
            for out, label in zip(outputs_np, labels_np):
                pred_alleles = []
                true_alleles = []
                for i, locus in enumerate(gene_order):
                    # 提取每个基因座的4个值（a1, a2, b1, b2）
                    start = i * 4
                    end = start + 4
                    pred = out[start:end]
                    true = label[start:end]

                    # 处理AMEL基因座（数值转字符）
                    if locus == "AMEL":
                        pred_char = [['X', 'Y'][int(round(x))] for x in pred]
                        true_char = [['X', 'Y'][int(round(x))] for x in true]
                        pred_alleles.extend(pred_char)
                        true_alleles.extend(true_char)
                    else:
                        # 数值型基因座：四舍五入为整数
                        pred_int = np.round(pred).astype(int)
                        true_int = np.round(true).astype(int)
                        pred_alleles.extend(pred_int)
                        true_alleles.extend(true_int)

                # 计算等位基因准确率
                correct = np.sum(np.array(pred_alleles) == np.array(true_alleles))
                total_correct += correct
                total_alleles += len(pred_alleles)

                predictions.append(pred_alleles)
                truths.append(true_alleles)

            # 计算MAE
            mae = torch.mean(torch.abs(outputs - labels)).item()
            total_mae += mae * labels.size(0)
            total_samples += labels.size(0)

    # 输出准确率
    allele_accuracy = total_correct / total_alleles if total_alleles != 0 else 0.0
    print(f'等位基因准确率: {allele_accuracy * 100:.2f}%')

    # 输出前5个样本的真实值与预测值对比（示例）
    print("\n真实值与预测值对比（前5个样本）:")
    for i in range(min(5, len(predictions))):
        print(f"样本 {i + 1}:")
        print(f"  真实值: {truths[i]}")
        print(f"  预测值: {predictions[i]}")
        print()

    return total_mae / total_samples


# ----------------------
# 5. 主函数（执行流程）
# ----------------------
def main():
    TRUTH_FILE = 'D题附件：法医物证多人身份鉴定问题数据集/附件3：各个贡献者对应的基因型数据.xlsx'  # 真值文件：Sample ID + 16基因座列（AM为X/Y）
    DATA_FILE = 'Data_T1.xlsx'  # 数据文件：每16行一个样本，包含Marker/Size/Height等列

    dataset = STRDataset(DATA_FILE, TRUTH_FILE)

    # 使用20%数据作为测试集（原逻辑为80%训练集，20%测试集）
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = STRGenotypeCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_model(train_loader, model, criterion, optimizer)
    # 传递truth_dict和gene_order参数
    evaluate_model(test_loader, model, dataset.truth_dict, GENE_ORDER)
    torch.save(model.state_dict(), 'str_genotype_model.pth')


if __name__ == '__main__':
    main()