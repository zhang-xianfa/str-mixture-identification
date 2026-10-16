import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.utils import resample
import xgboost as xgb
from tqdm import tqdm
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


def load_and_preprocess(file_path):
    """数据加载和预处理"""
    data = pd.read_excel(file_path)

    # 等位基因编码处理
    for i in range(1, 15):
        data[f'Allele {i}'] = data[f'Allele {i}'].apply(
            lambda x: 99 if x == 'X' else (100 if x == 'Y' else x)
        )

    # 处理可能的NaN值
    for col in data.columns:
        if data[col].dtype == 'float':
            data[col] = data[col].fillna(0)

    return data


def manual_oversample(X, y):
    """手动实现过采样"""
    max_samples = y.value_counts().max()
    balanced_X = []
    balanced_y = []

    for class_label in y.unique():
        X_class = X[y == class_label]
        y_class = y[y == class_label]

        X_resampled = resample(X_class,
                               replace=True,
                               n_samples=max_samples,
                               random_state=42)
        y_resampled = pd.Series([class_label] * max_samples)

        balanced_X.append(X_resampled)
        balanced_y.append(y_resampled)

    return pd.concat(balanced_X), pd.concat(balanced_y)


def extract_features(data):
    """特征提取主函数（完全按照extract_features_for_sample实现）"""
    X_samples = []
    y_samples = []

    print("\n正在提取特征...")
    for i in tqdm(range(0, len(data), 16)):
        sample_data = data.iloc[i:i + 16]
        label = sample_data.iloc[0]['label']

        # 完全按照extract_features_for_sample实现
        features = {}

        # 收集所有峰的信息
        all_heights = []
        all_sizes = []
        all_alleles = []

        for _, row in sample_data.iterrows():
            heights = row[[f'Height {i}' for i in range(1, 15)]].values
            sizes = row[[f'Size {i}' for i in range(1, 15)]].values
            alleles = row[[f'Allele {i}' for i in range(1, 15)]].values

            all_heights.extend(heights)
            all_sizes.extend(sizes)
            all_alleles.extend(alleles)

        # 转换为numpy数组
        all_heights = np.array(all_heights)
        all_sizes = np.array(all_sizes)
        all_alleles = np.array(all_alleles)

        # 计算动态阈值(均值高度)
        mean_height = np.mean(all_heights[all_heights > 0]) if np.any(all_heights > 0) else 0

        # 有效峰(高度>阈值)
        valid_mask = (all_heights > mean_height) & (all_heights > 0)
        valid_heights = all_heights[valid_mask]
        valid_sizes = all_sizes[valid_mask]
        valid_alleles = all_alleles[valid_mask]

        # 1. 基本峰特征
        features['total_peaks'] = np.sum(all_heights > 0)
        features['valid_peaks'] = len(valid_heights)
        features['valid_peak_ratio'] = features['valid_peaks'] / features['total_peaks'] if features[
                                                                                                'total_peaks'] > 0 else 0

        # 2. 峰高统计特征
        if len(valid_heights) > 0:
            features['mean_height'] = np.mean(valid_heights)
            features['max_height'] = np.max(valid_heights)
            features['height_std'] = np.std(valid_heights)
            features['height_kurtosis'] = pd.Series(valid_heights).kurt()

            # 峰高比例特征
            sorted_h = np.sort(valid_heights)[::-1]
            if len(sorted_h) >= 2:
                features['peak_ratio_1v2'] = sorted_h[0] / sorted_h[1]
                features['peak_ratio_1v3'] = sorted_h[0] / sorted_h[2] if len(sorted_h) >= 3 else 0
                features['peak_ratio_2v3'] = sorted_h[1] / sorted_h[2] if len(sorted_h) >= 3 else 0

        # 3. 片段大小特征
        if len(valid_sizes) > 0:
            features['mean_size'] = np.mean(valid_sizes)
            features['size_range'] = np.max(valid_sizes) - np.min(valid_sizes)
            features['size_std'] = np.std(valid_sizes)

        # 4. 等位基因特征
        features['unique_alleles'] = len(np.unique(valid_alleles)) if len(valid_alleles) > 0 else 0
        features['has_XY'] = int(np.isin([99, 100], valid_alleles).any())

        # 5. 按基因座统计的特征
        per_locus_features = []
        for _, row in sample_data.iterrows():
            locus_heights = row[[f'Height {i}' for i in range(1, 15)]].values
            locus_sizes = row[[f'Size {i}' for i in range(1, 15)]].values

            locus_valid = (locus_heights > mean_height) & (locus_heights > 0)
            valid_h = locus_heights[locus_valid]
            valid_s = locus_sizes[locus_valid]

            if len(valid_h) > 0:
                locus_feat = {
                    'locus_peaks': len(valid_h),
                    'locus_height_ratio': valid_h[0] / valid_h[1] if len(valid_h) >= 2 else 0,
                    'locus_size_range': np.max(valid_s) - np.min(valid_s) if len(valid_s) > 0 else 0
                }
                per_locus_features.append(locus_feat)

        # 添加平均基因座特征
        if per_locus_features:
            avg_locus = pd.DataFrame(per_locus_features).mean().to_dict()
            features.update({f'avg_locus_{k}': v for k, v in avg_locus.items()})

        X_samples.append(features)
        y_samples.append(label - 2)

    return pd.DataFrame(X_samples).fillna(0), pd.Series(y_samples)


def main():
    # 加载数据
    data = load_and_preprocess('Data_T1.xlsx')

    # 特征提取
    X, y = extract_features(data)

    # 数据检查
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(X)}")
    print("各类别分布:")
    print(y.value_counts().sort_index())

    # 数据平衡处理
    X_res, y_res = manual_oversample(X, y)
    print("\n=== 平衡后分布 ===")
    print(y_res.value_counts().sort_index())

    # 模型训练与评估
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("\n=== 交叉验证评估 ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_res, y_res, cv=cv)
    print(classification_report(y_res, y_pred))

    # 特征重要性
    model.fit(X_res, y_res)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n=== Top 10重要特征 ===")
    print(importance.head(10))


if __name__ == "__main__":
    main()
