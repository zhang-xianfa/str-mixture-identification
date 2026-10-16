import pandas as pd

df = pd.read_excel(r"..\..\data\data1_diff_nums.xlsx")
processed = []

for _, row in df.iterrows():
    row_data = row.tolist()
    processed_row = []

    processed_row.extend(row_data[:2])
    i = 3
    while i < len(row_data):
        if 'Allele' in df.columns[i]:
            allele = row_data[i]

            if allele == 'OL':
                i += 3
                continue

            if pd.isna(allele):
                break
            processed_row.extend(row_data[i:i+3])
            i += 3
    processed.append(processed_row)

max_col = max(len(row) for row in processed)
new_col = ['Sample File', 'Marker']
count = 1
for i in range(2, max_col, 3):
    new_col.extend([f'Allele {count}', f'Size {count}', f'Height {count}'])
    count += 1

processed_df = pd.DataFrame(processed, columns=new_col)
processed_df.to_excel('Data_T1.xlsx',  index=False)


# 可视化
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import numpy as np


def visualize_str_data(file_path):
    # 读取数据
    processed_df = pd.read_excel(file_path)

    # 1. 样本-标记热力图（Allele Presence）
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        processed_df.iloc[:, 2::3].notna(),
        cmap='Blues',
        cbar=False,
        yticklabels=processed_df['Sample File'].values
    )
    plt.title("Allele Presence by Sample (White=Missing)")
    plt.xlabel("Allele Index")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.show()

    # 2. 标记检出率雷达图
    markers = processed_df['Marker'].unique()
    detection_rates = [
        processed_df[processed_df['Marker'] == m].iloc[:, 2::3].notna().mean().mean()
        for m in markers
    ]

    angles = np.linspace(0, 2 * np.pi, len(markers), endpoint=False).tolist()
    angles += angles[:1]
    detection_rates += detection_rates[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, detection_rates, 'o-', linewidth=2)
    ax.fill(angles, detection_rates, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(markers)
    ax.set_title("Marker Detection Rate", pad=20)
    plt.show()

    # 3. 峰高（Height）分布
    height_cols = [c for c in processed_df.columns if 'Height' in c]
    all_heights = processed_df[height_cols].values.flatten()
    all_heights = all_heights[~pd.isna(all_heights)]

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    sns.boxplot(y=all_heights)
    plt.ylabel("RFU Height")
    plt.subplot(122)
    sns.histplot(all_heights, bins=30, kde=True)
    plt.xlabel("RFU Height")
    plt.suptitle("Peak Height Distribution")
    plt.tight_layout()
    plt.show()

    # 4. Size分布（按标记分组）
    size_cols = [c for c in processed_df.columns if 'Size' in c]
    size_data = []
    for m in markers:
        m_sizes = processed_df[processed_df['Marker'] == m][size_cols].values.flatten()
        size_data.append(m_sizes[~np.isnan(m_sizes)])

    plt.figure(figsize=(12, 5))
    sns.violinplot(data=size_data, inner="quartile")
    plt.xticks(range(len(markers)), markers, rotation=45)
    plt.ylabel("Basepair Size")
    plt.title("Allele Size Distribution by Marker")
    plt.tight_layout()
    plt.show()

    # 5. 数据质量报告
    allele_cols = [c for c in processed_df.columns if 'Allele' in c]
    stats = {
        "Total Samples": len(processed_df),
        "Mean Alleles/Sample": processed_df[allele_cols].notna().sum(axis=1).mean(),
        "Median Height (RFU)": np.median(all_heights),
        "Max Missing Rate": (1 - processed_df[allele_cols].notna().mean().max()) * 100
    }

    print("\n=== Data Quality Report ===")
    for k, v in stats.items():
        print(f"{k:<20}: {v:.1f}" if isinstance(v, float) else f"{k:<20}: {v}")


# 使用示例
if __name__ == "__main__":
    visualize_str_data("Data_T1.xlsx")
