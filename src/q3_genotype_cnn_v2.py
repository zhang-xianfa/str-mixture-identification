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
LR = 1e-4
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
# 2. 自定义数据集类（3人混合样本，通过num==3筛选）
# ----------------------
class STRDataset3Contrib(Dataset):
    def __init__(self, data_file, truth_file, max_size=MAX_SIZE):
        self.data_df = pd.read_excel(data_file)
        self.truth_dict = load_ground_truth(truth_file)
        self.max_size = max_size
        self.scaler = StandardScaler()

        # 筛选num==3的样本
        self.filtered_df = self._filter_samples_by_num()

        # 验证数据行数是否为16的倍数
        if len(self.filtered_df) % NUM_GENE_LOCI != 0:
            raise ValueError("筛选后的数据行数必须为16的倍数（每个样本16行）")

        # 按每16行分组，每组为一个混合样本
        self.samples = [
            self.filtered_df.iloc[i:i + NUM_GENE_LOCI]
            for i in range(0, len(self.filtered_df), NUM_GENE_LOCI)
        ]

        # 检查数据列是否包含必要字段
        required_cols = ['num', 'Marker', 'Size 1', 'Height 1', 'Size 2', 'Height 2']
        if not set(required_cols).issubset(self.filtered_df.columns):
            raise ValueError("数据文件缺少必要列：num, Marker, Size 1, Height 1, Size 2, Height 2")

        self.locus_matrix, self.labels = self._preprocess_data()

    def _filter_samples_by_num(self):
        """通过num列筛选3人混合样本"""
        # 按Sample File分组，每组的num应该相同
        grouped = self.data_df.groupby('Sample File')

        valid_samples = []
        for name, group in grouped:
            # 获取该样本的num值
            sample_num = group['num'].iloc[0]

            # 只保留num==3的样本
            if sample_num == 3:
                valid_samples.append(group)

        if not valid_samples:
            raise ValueError("数据集中未找到num==3的样本")

        filtered_df = pd.concat(valid_samples)
        print(f"从{len(grouped)}个样本中筛选出{len(valid_samples)}个num==3的样本")
        return filtered_df

    def _preprocess_data(self):
        """预处理数据并确保所有标签都是96维"""
        matrix_list = []
        labels_list = []

        for sample_group in self.samples:
            # 解析样本ID，确定贡献者数量
            sample_file = sample_group.iloc[0]['Sample File']
            if pd.isna(sample_file):
                raise ValueError("Sample File列为空")

            try:
                # 解析贡献者ID
                parts = sample_file.split('-')
                if len(parts) < 3:
                    raise ValueError("Sample File格式错误，缺少ID部分")
                ids_part = parts[2]  # 例如："44_45_46"
                contrib_ids = list(map(int, ids_part.split('_')))

                # 确保是3个贡献者
                if len(contrib_ids) != 3:
                    print(f"警告：样本 {sample_file} 的Sample File包含 {len(contrib_ids)} 个贡献者，与num=3矛盾，跳过")
                    continue

            except Exception as e:
                print(f"样本ID解析失败：{sample_file}，错误：{e}，跳过")
                continue

            # 验证贡献者ID是否存在于真值文件
            if any(contrib_id not in self.truth_dict for contrib_id in contrib_ids):
                missing = [cid for cid in contrib_ids if cid not in self.truth_dict]
                print(f"警告：贡献者ID不存在于真值文件：{missing}，跳过样本 {sample_file}")
                continue

            # 处理峰高数据
            sample_matrix = np.zeros((NUM_GENE_LOCI, self.max_size))

            for row_idx, (_, row) in enumerate(sample_group.iterrows()):
                # 基因座名称处理
                raw_marker = str(row['Marker'])
                processed_marker = raw_marker.replace(" ", "").upper()
                expected_marker = GENE_ORDER[row_idx].replace(" ", "").upper()

                if not processed_marker.startswith(expected_marker):
                    print(f"警告：基因座匹配失败: {raw_marker} vs {expected_marker}，跳过样本 {sample_file}")
                    sample_matrix = None
                    break

                # 解析等位基因
                size1 = row['Size 1']
                height1 = row['Height 1'] if not pd.isna(row['Height 1']) else 0.0
                size2 = row['Size 2']
                height2 = row['Height 2'] if not pd.isna(row['Height 2']) else 0.0

                if 1 <= size1 <= self.max_size:
                    sample_matrix[row_idx, int(size1) - 1] += height1
                if 1 <= size2 <= self.max_size:
                    sample_matrix[row_idx, int(size2) - 1] += height2

            # 如果处理过程中出错，跳过该样本
            if sample_matrix is None:
                continue

            # 标准化处理
            for col in range(self.max_size):
                col_data = sample_matrix[:, col].reshape(-1, 1)
                if np.any(col_data != 0):
                    sample_matrix[:, col] = self.scaler.fit_transform(col_data).flatten()

            # 生成标签
            label = []
            for locus in GENE_ORDER:
                for contrib_id in contrib_ids:
                    alleles = self.truth_dict[contrib_id][locus]
                    if locus == "AMEL":
                        # AMEL基因座转为数值（X=0, Y=1）
                        a1 = 0 if alleles[0] == "X" else 1
                        a2 = 0 if alleles[1] == "X" else 1
                        label.extend([a1, a2])
                    else:
                        # 数值型基因座直接添加
                        label.extend([alleles[0], alleles[1]])

            matrix_list.append(sample_matrix)
            labels_list.append(np.array(label, dtype=np.float32))

        # 检查所有标签是否为96维
        for i, lbl in enumerate(labels_list):
            if len(lbl) != 96:
                print(f"警告：样本 {i} 的标签长度为 {len(lbl)}，应为96，跳过")
                labels_list[i] = None
                matrix_list[i] = None

        # 移除无效样本
        matrix_list = [m for m in matrix_list if m is not None]
        labels_list = [l for l in labels_list if l is not None]

        if not matrix_list or not labels_list:
            raise ValueError("经过筛选后，没有有效的3人混合样本数据")

        return np.array(matrix_list), np.array(labels_list)

    def __len__(self):
        return len(self.locus_matrix)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.locus_matrix[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ----------------------
# 3. CNN模型定义（3人混合样本）
# ----------------------
class STRGenotypeCNN3Contrib(nn.Module):
    def __init__(self, input_channels=NUM_GENE_LOCI, output_dim=96):
        super().__init__()
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
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ----------------------
# 4. 训练函数
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


# ----------------------
# 5. 评估函数（3人混合样本）
# ----------------------
def evaluate_model_3contrib(test_loader, model, truth_dict, gene_order):
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
                    # 提取每个基因座的6个值（3个人，每人2个等位基因）
                    start = i * 6
                    end = start + 6
                    pred = out[start:end]
                    true = label[start:end]

                    # 处理AMEL基因座
                    if locus == "AMEL":
                        pred_char = [['X', 'Y'][int(round(x))] for x in pred]
                        true_char = [['X', 'Y'][int(round(x))] for x in true]
                        pred_alleles.extend(pred_char)
                        true_alleles.extend(true_char)
                    else:
                        # 数值型基因座
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
    print(f'3人混合样本等位基因准确率: {allele_accuracy * 100:.2f}%')

    # 输出前5个样本的真实值与预测值对比
    print("\n3人混合样本真实值与预测值对比（前5个样本）:")
    for i in range(min(5, len(predictions))):
        print(f"样本 {i + 1}:")
        print(f"  真实值: {truths[i]}")
        print(f"  预测值: {predictions[i]}")
        print()

    return total_mae / total_samples


# ----------------------
# 6. 预测函数（3人混合样本）
# ----------------------
def predict_3contrib(model_path, data_file, truth_file):
    """预测3人混合样本"""
    # 加载3人模型
    model = STRGenotypeCNN3Contrib()
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    # 创建3人数据集
    dataset = STRDataset3Contrib(data_file, truth_file)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 预测并输出结果
    all_predictions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(DEVICE)
            outputs = model(data)
            outputs_np = outputs.cpu().numpy()

            # 解析预测结果
            for out in outputs_np:
                pred_alleles = []
                for i, locus in enumerate(GENE_ORDER):
                    start = i * 6
                    end = start + 6
                    pred = out[start:end]

                    if locus == "AMEL":
                        pred_char = [['X', 'Y'][int(round(x))] for x in pred]
                        pred_alleles.extend(pred_char)
                    else:
                        pred_int = np.round(pred).astype(int)
                        pred_alleles.extend(pred_int)

                all_predictions.append(pred_alleles)

    # 格式化输出
    print("\n3人混合样本预测结果:")
    for i, pred in enumerate(all_predictions):
        print(f"样本 {i + 1}:")
        print(f"  预测基因型: {pred}")
        print()

    return all_predictions


# ----------------------
# 7. 主函数
# ----------------------
def main():
    # 配置文件路径
    TRUTH_FILE = 'D题附件：法医物证多人身份鉴定问题数据集/附件3：各个贡献者对应的基因型数据.xlsx'
    DATA_FILE = 'Data_T1.xlsx'  # 包含2-5人混合样本的数据
    MODEL_PATH = 'str_genotype_model_3contrib.pth'

    # 创建数据集（自动筛选num==3的样本）
    print("正在加载数据并筛选num==3的3人混合样本...")
    dataset = STRDataset3Contrib(DATA_FILE, TRUTH_FILE)

    print(f"成功加载 {len(dataset)} 个符合条件的3人混合样本")

    # 划分训练集和测试集（80%训练，20%测试）
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    model = STRGenotypeCNN3Contrib()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练模型
    print("开始训练3人混合样本模型...")
    train_model(train_loader, model, criterion, optimizer)

    # 评估模型
    print("开始评估3人混合样本模型...")
    evaluate_model_3contrib(test_loader, model, dataset.truth_dict, GENE_ORDER)

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至: {MODEL_PATH}")

    # 使用模型进行预测（可选）
    print("开始使用模型进行预测...")
    predict_3contrib(MODEL_PATH, DATA_FILE, TRUTH_FILE)


if __name__ == '__main__':
    main()