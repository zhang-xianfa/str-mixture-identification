import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel('Data_T1.xlsx')

for i in range(1, 15):
    data[f'Allele {i}'] = data[f'Allele {i}'].apply(
        lambda x: 99 if x == 'X' else (100 if x == 'Y' else x)
    )

features = []
for i in range(1, 15):
    features.extend([f'Allele {i}', f'Size {i}', f'Height {i}'])
X = data[features].fillna(0)
y = LabelEncoder().fit_transform(data['label'])


def add_forensic_features(df):
    df['valid_peaks_count'] = 0
    df['dynamic_peak_ratio'] = 0.0
    df['peak_ratio_1v2'] = 0.0
    df['size_variation'] = 0.0
    df['height_kurtosis'] = 0.0
    df['has_XY'] = 0

    for idx, row in df.iterrows():
        # 提取当前样本的所有峰高度
        heights = row[[f'Height {i}' for i in range(1, 15)]].values
        sizes = row[[f'Size {i}' for i in range(1, 15)]].values
        alleles = row[[f'Allele {i}' for i in range(1, 15)]].values

        # 计算动态阈值（均值）
        mean_height = np.mean(heights)

        # 筛选有效峰（高度 > 均值）
        valid_mask = heights > mean_height
        valid_heights = heights[valid_mask]
        valid_sizes = sizes[valid_mask]
        valid_alleles = alleles[valid_mask]

        # 保存有效峰数量
        df.at[idx, 'valid_peaks_count'] = len(valid_heights)

        # 1. 峰高比特征
        if len(valid_heights) >= 2:
            sorted_h = np.sort(valid_heights)[::-1]
            df.at[idx, 'dynamic_peak_ratio'] = sorted_h[-1] / sorted_h[0]  # 最小峰高比
            df.at[idx, 'peak_ratio_1v2'] = sorted_h[0] / sorted_h[1]  # 主次峰比

        # 2. 等位基因间距一致性
        if len(valid_sizes) > 0:
            df.at[idx, 'size_variation'] = np.std(valid_sizes)

        # 3. 高度分布峰度
        if len(valid_heights) > 0:
            df.at[idx, 'height_kurtosis'] = pd.Series(valid_heights).kurt()

        # 4. 性别染色体标记
        df.at[idx, 'has_XY'] = int(any(a in [99, 100] for a in valid_alleles))

    return df

X = add_forensic_features(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    n_estimators=800,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    max_bin=1024,
    reg_alpha=0.2,
    reg_lambda=0.2,
    tree_method='hist',
    enable_categorical=True,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop Features:\n", importance)
