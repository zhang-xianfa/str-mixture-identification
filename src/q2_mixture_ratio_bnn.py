import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
# 设置随机种子
torch.manual_seed(42)
pyro.set_rng_seed(42)
print("1. 已设置随机种子")

GENE_LOCI = [
    "D8S1179", "D21S11", "D7S820", "CSF1PO",
    "D3S1358", "TH01", "D13S317", "D16S539",
    "D2S1338", "D19S433", "vWA", "TPOX",
    "D18S51", "AMEL", "D5S818", "FGA"
]


# ----------------------------
# 1. 数据预处理（支持多人数混合）
# ----------------------------
def load_and_preprocess_data(filepath):
    print("\n2. 开始数据预处理...")
    df = pd.read_excel(filepath)
    print(f"原始数据行数: {len(df)}")

    df['num'] = df['num'].astype(int)
    df_combined = df.copy()

    def extract_ratio_with_padding(sample_str, num_contributors):
        try:
            parts = sample_str.split('-')
            ratio_part = parts[3].replace(';', ':')
            ratios = list(map(float, ratio_part.split(':')))

            # 确保提取的比例数量与标注的人数一致
            if len(ratios) != num_contributors:
                print(f"警告: 样本 {sample_str} 提取的比例数({len(ratios)})与标注人数({num_contributors})不匹配")
                # 尝试修复：如果比例数少于人数，补零
                if len(ratios) < num_contributors:
                    print(f"尝试修复: 为样本 {sample_str} 补零至 {num_contributors} 个比例")
                    ratios = ratios + [0.0] * (num_contributors - len(ratios))
                # 如果比例数多于人数，截断
                else:
                    print(f"尝试修复: 为样本 {sample_str} 截断至 {num_contributors} 个比例")
                    ratios = ratios[:num_contributors]

            normalized = [r / sum(ratios) for r in ratios] if sum(ratios) != 0 else [0.0] * num_contributors
            return normalized
        except Exception as e:
            print(f"错误: 处理样本 {sample_str} 时出错: {e}")
            return [0.0] * num_contributors

    df_combined['ratio'] = df_combined.apply(
        lambda x: extract_ratio_with_padding(x['Sample File'], x['num']), axis=1
    )

    print("比例提取完成，示例如下:")
    print(df_combined[['Sample File', 'num', 'ratio']].head())
    return df_combined


# ----------------------------
# 2. 增强版特征工程（统一特征维度）
# ----------------------------
def build_features(df, max_alleles=14, feature_length=60):
    print("\n3. 开始特征工程...")
    features = []
    valid_samples = []
    skipped_samples = 0

    df = df.sort_values(['Sample File', 'Marker'])
    total_samples = len(df) // 16
    print(f"预期处理样本数: {total_samples}")

    for i in range(total_samples):
        start_idx = i * 16
        end_idx = start_idx + 16
        if end_idx > len(df):
            skipped_samples += 1
            continue

        sample_data = df.iloc[start_idx:end_idx]
        sample_id = sample_data['Sample File'].iloc[0]
        num_contributors = sample_data['num'].iloc[0]
        ratio = sample_data['ratio'].iloc[0]

        all_heights = []
        all_sizes = []
        for _, row in sample_data.iterrows():
            for j in range(1, max_alleles + 1):
                h = row.get(f'Height {j}', np.nan)
                s = row.get(f'Size {j}', np.nan)
                if pd.notna(h) and pd.notna(s):
                    all_heights.append(h)
                    all_sizes.append(s)

        if len(all_heights) < 2:
            skipped_samples += 1
            continue

        feature = np.zeros(feature_length, dtype=np.float32)
        valid_peaks = sorted([(s, h) for s, h in zip(all_sizes, all_heights) if h > 0], key=lambda x: -x[1])

        feature[0] = len(valid_peaks)
        top_peaks = valid_peaks[:min(4, len(valid_peaks))]

        for idx, (s, h) in enumerate(top_peaks):
            feature[1 + 2 * idx] = s
            feature[2 + 2 * idx] = h

        for i in range(4):
            for j in range(i + 1, 4):
                if i < len(top_peaks) and j < len(top_peaks):
                    feature[9 + i * 4 + j] = top_peaks[i][1] / top_peaks[j][1]

        if all_heights:
            feature[25] = np.mean(all_heights)
            feature[26] = np.std(all_heights)

        features.append(feature)
        valid_samples.append({
            'Sample File': sample_id,
            'ratio': ratio,
            'num': num_contributors
        })

    features_array = np.vstack(features)
    df_valid = pd.DataFrame(valid_samples).reset_index(drop=True)

    print(f"\n特征工程完成:")
    print(f"- 总样本数: {total_samples}")
    print(f"- 有效样本数: {len(df_valid)}")
    print(f"- 特征矩阵形状: {features_array.shape}")
    return features_array, df_valid


# ----------------------------
# 3. 动态贝叶斯神经网络模型
# ----------------------------
class DynamicBayesianNN(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 128)
        self.fc1.weight = PyroSample(dist.Normal(0, 0.1).expand([128, input_dim]).to_event(2))

        self.fc2 = PyroModule[nn.Linear](128, 64)

        # 为不同人数创建独立输出层
        self.out2 = PyroModule[nn.Linear](64, 2)  # 2人混合
        self.out3 = PyroModule[nn.Linear](64, 3)  # 3人混合
        self.out4 = PyroModule[nn.Linear](64, 4)  # 4人混合
        self.out5 = PyroModule[nn.Linear](64, 5)  # 5人混合

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, num_contributors, y=None):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        # 根据人数选择输出层
        if num_contributors == 2:
            probs = torch.sigmoid(self.out2(x))
        elif num_contributors == 3:
            probs = torch.sigmoid(self.out3(x))
        elif num_contributors == 4:
            probs = torch.sigmoid(self.out4(x))
        elif num_contributors == 5:
            probs = torch.sigmoid(self.out5(x))
        else:
            raise ValueError(f"不支持的贡献者数量: {num_contributors}")

        # 归一化，确保比例和为1
        probs = probs / torch.sum(probs, dim=1, keepdim=True)

        with pyro.plate("data", x.shape[0]):
            if y is not None:
                for i in range(probs.shape[1]):
                    pyro.sample(f"obs{i + 1}", dist.Normal(probs[:, i], 0.05), obs=y[:, i])
        return probs


# ----------------------------
# 4. 训练和评估函数
# ----------------------------
def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs):
    print("\n5. 开始训练...")
    pyro.clear_param_store()
    svi = pyro.infer.SVI(
        model=model,
        guide=pyro.infer.autoguide.AutoNormal(model),
        optim=optimizer,
        loss=pyro.infer.Trace_ELBO()
    )

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y, num_contributors) in enumerate(train_loader):
            # 按人数分组计算损失
            for nc in torch.unique(num_contributors):
                mask = (num_contributors == nc)
                x_batch = x[mask]
                y_batch = [y[i] for i, m in enumerate(mask) if m]
                if len(x_batch) == 0:
                    continue

                # 将标签列表转换为张量
                y_tensor = torch.stack(y_batch)

                loss = svi.step(x_batch, nc.item(), y_tensor)
                total_loss += loss

            if batch_idx == 0 and epoch % 100 == 0:
                with torch.no_grad():
                    sample_nc = num_contributors[0].item()
                    sample_pred = model(x[:1], sample_nc)
                    print(
                        f"Epoch {epoch} 首批次示例 - 输入: {x[0][:5]}..., "
                        f"预测: {sample_pred.numpy()}, 真实: {y[0].numpy()}, 人数: {sample_nc}"
                    )

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")

    print("\n6. 开始评估...")
    model.eval()
    test_preds, test_true, test_nums = [], [], []
    with torch.no_grad():
        for batch_idx, (x, y, num_contributors) in enumerate(test_loader):
            for nc in torch.unique(num_contributors):
                mask = (num_contributors == nc)
                x_batch = x[mask]
                y_batch = [y[i] for i, m in enumerate(mask) if m]
                if len(x_batch) == 0:
                    continue

                pred = model(x_batch, nc.item())
                test_preds.extend([p for p in pred])  # 保存为列表而不是拼接
                test_true.extend(y_batch)
                test_nums.append(torch.full_like(num_contributors[mask], nc))

            if batch_idx == 0:
                print("\n测试集首批次示例:")
                for i in range(min(3, len(x))):
                    print(f"样本{i}: 输入={x[i][:5]}..., 预测={model(x[i:i + 1], num_contributors[i].item()).numpy()}, "
                          f"真实={y[i].numpy()}, 人数={num_contributors[i].item()}")

    test_nums = torch.cat(test_nums).numpy()

    # 将预测和真实标签转换为固定长度的数组（最多5个组分）
    max_components = 5
    test_preds_padded = np.zeros((len(test_preds), max_components))
    test_true_padded = np.zeros((len(test_true), max_components))

    for i, (pred, true, nc) in enumerate(zip(test_preds, test_true, test_nums)):
        pred_np = pred.numpy()
        true_np = true.numpy()
        test_preds_padded[i, :int(nc)] = pred_np
        test_true_padded[i, :int(nc)] = true_np

    # 计算各组分MAE
    mae_by_num = {}
    for nc in np.unique(test_nums):
        mask = (test_nums == nc)
        nc_preds = test_preds_padded[mask, :int(nc)]
        nc_true = test_true_padded[mask, :int(nc)]
        nc_mae = mean_absolute_error(nc_true, nc_preds)
        mae_by_num[nc] = nc_mae
        print(f"{int(nc)}人混合样本MAE: {nc_mae:.4f}")

    overall_mae = mean_absolute_error(test_true_padded, test_preds_padded)
    print(f"\n整体MAE: {overall_mae:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(15, 10))
    for nc in np.unique(test_nums):
        mask = (test_nums == nc)
        plt.subplot(2, 2, nc - 1)
        for i in range(int(nc)):
            plt.scatter(test_true_padded[mask, i], test_preds_padded[mask, i], alpha=0.5, label=f'组分{i + 1}')
        plt.plot([0, 1], [0, 1], 'r--', label='理想预测')
        plt.xlabel('真实比例')
        plt.ylabel('预测比例')
        plt.title(f'{int(nc)}人混合样本预测结果')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multi_component_prediction.png')
    print("\n已保存多组分预测可视化图表")

    # 创建结果DataFrame
    results_data = {'样本ID': range(len(test_preds)), '实际人数': test_nums}
    for i in range(max_components):
        results_data[f'真实比例_组分{i + 1}'] = test_true_padded[:, i]
        results_data[f'预测比例_组分{i + 1}'] = test_preds_padded[:, i]

    results_df = pd.DataFrame(results_data)
    results_df.to_csv('results/multi_component_predictions.csv', index=False)
    print("已保存全组分预测结果到CSV文件")

    return overall_mae, mae_by_num, train_losses, results_df


# ----------------------------
# 5. 主训练流程
# ----------------------------
def main():
    print("\n=== 多人数DNA混合物比例预测模型 ===")
    # 参数设置
    DATA_PATH = 'Data_T2.xlsx'
    MAX_ALLELES = 14
    TEST_SIZE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 2000
    FEATURE_LENGTH = 60

    # 1. 加载和预处理数据
    df = load_and_preprocess_data(DATA_PATH)

    # 2. 构建特征和标签
    X, df_valid = build_features(df, max_alleles=MAX_ALLELES, feature_length=FEATURE_LENGTH)

    # 详细检查标签长度
    print("\n详细标签检查:")
    for i, row in df_valid.iterrows():
        if len(row['ratio']) != row['num']:
            print(f"样本 {i}: 文件={row['Sample File']}, 标注人数={row['num']}, 标签长度={len(row['ratio'])}")
            print(f"标签内容: {row['ratio']}")

    # 验证所有标签长度是否正确
    invalid_samples = df_valid[df_valid.apply(lambda x: len(x['ratio']) != x['num'], axis=1)]
    if not invalid_samples.empty:
        print(f"\n警告：发现 {len(invalid_samples)} 个样本的标签长度与人数不匹配")
        print("尝试自动修复...")

        # 自动修复标签长度
        df_valid['ratio'] = df_valid.apply(
            lambda x: x['ratio'] + [0.0] * (x['num'] - len(x['ratio'])) if len(x['ratio']) < x['num']
            else x['ratio'][:x['num']],
            axis=1
        )

        # 再次验证
        invalid_samples = df_valid[df_valid.apply(lambda x: len(x['ratio']) != x['num'], axis=1)]
        if not invalid_samples.empty:
            print(f"修复失败，仍有 {len(invalid_samples)} 个样本的标签长度与人数不匹配")
            print(invalid_samples[['Sample File', 'num', 'ratio']])
            raise ValueError("标签长度与人数不匹配，无法继续")
        else:
            print("所有样本标签已成功修复")

    print("\n标签验证完成")
    print("前5个样本的标签:")
    for i in range(5):
        print(f"样本{i}: 人数={df_valid['num'].iloc[i]}, 标签={df_valid['ratio'].iloc[i]}")

    # 3. 数据标准化
    print("\n进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = X.copy().astype(np.float32)
    X_scaled[:, 1:] = scaler.fit_transform(X[:, 1:])
    print("标准化后特征示例:")
    print(X_scaled[0, :10])

    # 4. 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    num_contributors_tensor = torch.tensor(df_valid['num'].values, dtype=torch.int32)

    # 创建可变长度标签张量列表
    y_tensors = []
    for i in range(len(df_valid)):
        y_tensors.append(torch.tensor(df_valid['ratio'].iloc[i], dtype=torch.float32))

    print("\n张量转换完成:")
    print(f"- 特征张量形状: {X_tensor.shape}")
    print(f"- 人数张量形状: {num_contributors_tensor.shape}")
    print(f"- 标签张量示例: {y_tensors[0]} (人数={num_contributors_tensor[0].item()})")

    # 5. 划分训练测试集
    indices = np.arange(len(X_tensor))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=42, stratify=num_contributors_tensor.numpy()
    )

    X_train, X_test = X_tensor[train_indices], X_tensor[test_indices]
    y_train, y_test = [y_tensors[i] for i in train_indices], [y_tensors[i] for i in test_indices]
    num_train, num_test = num_contributors_tensor[train_indices], num_contributors_tensor[test_indices]

    print("\n数据集划分完成:")
    print(f"- 训练集: {len(X_train)} 样本")
    print(f"- 测试集: {len(X_test)} 样本")
    print("训练集人数分布:", pd.Series(num_train.numpy()).value_counts())
    print("测试集人数分布:", pd.Series(num_test.numpy()).value_counts())

    # 6. 创建数据加载器
    class DNADataset(torch.utils.data.Dataset):
        def __init__(self, features, labels, num_contributors):
            self.features = features
            self.labels = labels
            self.num_contributors = num_contributors

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx], self.num_contributors[idx]

    train_dataset = DNADataset(X_train, y_train, num_train)
    test_dataset = DNADataset(X_test, y_test, num_test)

    # 自定义collate_fn处理可变长度标签
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        num_contributors = torch.stack([item[2] for item in batch])
        return features, labels, num_contributors

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    print("\n数据加载器创建完成")
    print(f"- 训练批次数量: {len(train_loader)}")
    print(f"- 测试批次数量: {len(test_loader)}")

    # 7. 创建模型
    model = DynamicBayesianNN(input_dim=FEATURE_LENGTH)

    # 8. 设置优化器
    optimizer = pyro.optim.Adam({"lr": 1e-4})
    print("\n优化器设置完成")

    # 9. 训练和评估模型
    overall_mae, mae_by_num, train_losses, results_df = train_and_evaluate(
        model, train_loader, test_loader, optimizer, num_epochs=EPOCHS
    )

    # 打印结果
    print("\n=== 最终结果 ===")
    for nc, mae in mae_by_num.items():
        print(f"{int(nc)}人混合样本MAE: {mae:.4f}")
    print(f"整体MAE: {overall_mae:.4f}")

    # 10. 保存模型
    torch.save(model.state_dict(), "dynamic_bayesian_model.pt")
    print("\n模型已保存")


if __name__ == "__main__":
    main()
