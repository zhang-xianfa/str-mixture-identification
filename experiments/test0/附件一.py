import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os for path joining
import re # Import re for regular expressions

# --- Configuration ---
# 请替换为您的实际文件路径
# 假设 "附件1：不同人数的STR图谱数据.xlsx" 位于 data 文件夹中
base_dir = r"E:\Program Files (x86)\pythonProject\数学建模\深圳杯" # 请根据您的实际项目根目录修改
data_dir = os.path.join(base_dir, "data")
annex_file_name = "data1_diff_nums.xlsx"
annex_file_path = os.path.join(data_dir, annex_file_name)

# --- Set Matplotlib style for better plots ---
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
# --- Add following lines for Chinese characters support ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # Add SimHei or other Chinese font
plt.rcParams['axes.unicode_minus'] = False # Solve the problem of displaying '-' as a square
# Set default figure size
plt.rcParams['figure.figsize'] = (10, 6)
# Set higher DPI for better quality images
plt.rcParams['figure.dpi'] = 100

# --- Function to parse Sample File name and extract NOC ---
def extract_noc_from_filename(filename):
    """
    Extracts the Number of Contributors (NOC) from the Sample File name.
    Assumes the format ...-ContributorIDs-Ratio-...
    ContributorIDs are separated by underscores.
    Example: A02_RD14-0003-40_41-1;4-M3S30-0.075IP-Q4.0_001.5sec.fsa -> IDs are 40_41 -> NOC = 2
    """
    match = re.search(r'RD14-0003-(\d+(_\d+)*)-', filename)
    if match:
        contributor_ids_str = match.group(1)
        # Count the number of IDs by splitting by underscore
        contributor_ids = contributor_ids_str.split('_')
        return len(contributor_ids)
    return None # Return None if pattern doesn't match

# --- 1. Load Data ---
print(f"正在从文件载入数据: {annex_file_path}")
try:
    df_annex1_wide = pd.read_excel(annex_file_path)
    print("数据载入成功。")
    print(f"原始数据形状 (宽格式): {df_annex1_wide.shape}")
    print("\n原始数据列信息:")
    # print(df_annex1_wide.info()) # info() might print too many columns
    print(f"列名: {df_annex1_wide.columns.tolist()}")
    print("\n原始数据前5行:")
    print(df_annex1_wide.head())

except FileNotFoundError:
    print(f"错误: 未找到文件，请检查路径是否正确: {annex_file_path}")
    exit()
except Exception as e:
    print(f"载入文件时发生错误: {e}")
    exit()

# --- Extract NOC before conversion (operate on sample files) ---
print("\n正在从样本文件名称中提取贡献者人数 (NOC)...")
df_annex1_wide['NOC'] = df_annex1_wide['Sample File'].apply(extract_noc_from_filename)

# Check if NOC extraction was successful for at least some rows
if df_annex1_wide['NOC'].isnull().all():
    print("警告: 未能从任何样本文件名称中提取到贡献者人数 (NOC)。请检查文件名格式是否符合预期。")
else:
    print("NOC 提取成功。前5行数据包含 NOC:")
    print(df_annex1_wide[['Sample File', 'NOC']].head().to_markdown(index=False, numalign="left", stralign="left"))
    print("\n提取的 NOC 分布:")
    print(df_annex1_wide['NOC'].value_counts().sort_index().to_markdown(numalign="left", stralign="left"))


# --- 2. Convert Wide Format to Long Format ---
print("\n正在将宽格式数据转换为长格式...")

# Identify the base columns and the repeating groups
# Keep NOC column as well
base_cols_long = ['Sample File', 'Marker', 'Dye', 'NOC']
allele_cols = [f'Allele {i}' for i in range(1, 101)]
size_cols = [f'Size {i}' for i in range(1, 101)]
height_cols = [f'Height {i}' for i in range(1, 101)]

# Check if required columns exist
required_base_cols = ['Sample File', 'Marker', 'Dye']
if not all(col in df_annex1_wide.columns for col in required_base_cols):
    missing = [col for col in required_base_cols if col not in df_annex1_wide.columns]
    print(f"错误: 原始数据缺少必需的基础列: {missing}")
    exit()

# Check if at least one set of allele/size/height columns exists
if not (allele_cols[0] in df_annex1_wide.columns and size_cols[0] in df_annex1_wide.columns and height_cols[0] in df_annex1_wide.columns):
     print("错误: 原始数据缺少必需的峰信息列 (如 Allele 1, Size 1, Height 1)。")
     exit()

# Stack the repeating groups
df_long_list = []

# Find the maximum index available in the columns
max_idx = 0
for col in df_annex1_wide.columns:
    if 'Allele ' in col:
        try:
            idx = int(col.split(' ')[1])
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            pass # Ignore columns that don't match the pattern

print(f"检测到最多 {max_idx} 组峰信息列。")

for i in range(1, max_idx + 1):
    a_col = f'Allele {i}'
    s_col = f'Size {i}'
    h_col = f'Height {i}'

    # Ensure columns exist before trying to select them
    current_cols = base_cols_long.copy()
    value_cols = []
    if a_col in df_annex1_wide.columns:
        current_cols.append(a_col)
        value_cols.append(a_col)
    if s_col in df_annex1_wide.columns:
        current_cols.append(s_col)
        value_cols.append(s_col)
    if h_col in df_annex1_wide.columns:
        current_cols.append(h_col)
        value_cols.append(h_col)

    if not value_cols: # Skip if none of the columns for this index exist
        continue

    # Select relevant columns
    # Use .reindex(columns=current_cols) to ensure all current_cols are present,
    # filling missing ones with NaN. This handles cases where some i columns are missing.
    df_temp = df_annex1_wide.reindex(columns=current_cols).copy()

    # Rename the value columns for stacking
    rename_map = {}
    if a_col in current_cols: rename_map[a_col] = 'Allele'
    if s_col in current_cols: rename_map[s_col] = 'Size'
    if h_col in current_cols: rename_map[h_col] = 'Height'

    df_temp.rename(columns=rename_map, inplace=True)

    df_long_list.append(df_temp)

# Concatenate all temporary dataframes
df_annex1_long = pd.concat(df_long_list, ignore_index=True)

# Drop rows where 'Allele' is NaN or empty string, as these are empty peak slots
initial_long_rows = df_annex1_long.shape[0]
df_annex1_long.dropna(subset=['Allele'], inplace=True)
df_annex1_long = df_annex1_long[df_annex1_long['Allele'] != ''].copy() # Also handle empty strings if any

print(f"转换为长格式后，有效峰数量: {df_annex1_long.shape[0]} (从 {initial_long_rows} 行中移除空峰)")
print("\n长格式数据列信息:")
print(df_annex1_long.info())
print("\n长格式数据前5行:")
print(df_annex1_long.head())

# Convert Size and Height to numeric, coercing errors to NaN
df_annex1_long['Size'] = pd.to_numeric(df_annex1_long['Size'], errors='coerce')
df_annex1_long['Height'] = pd.to_numeric(df_annex1_long['Height'], errors='coerce')

# Drop rows where Size or Height became NaN after coercion
rows_before_numeric_na_drop = df_annex1_long.shape[0]
df_annex1_long.dropna(subset=['Size', 'Height'], inplace=True)
if df_annex1_long.shape[0] < rows_before_numeric_na_drop:
    print(f"注意: 移除了 {rows_before_numeric_na_drop - df_annex1_long.shape[0]} 行由于 Height 或 Size 无法转为数值而丢失的峰。")


# --- 3. Overall Descriptive Analysis (Updated Terminology) ---
print("\n--- 总体描述性统计分析 ---")

# Basic counts
num_samples = df_annex1_long['Sample File'].nunique()
num_peaks = df_annex1_long.shape[0]
num_markers = df_annex1_long['Marker'].nunique()
num_dyes = df_annex1_long['Dye'].nunique()

print(f"总样本文件数: {num_samples}")
print(f"总峰数量 (有效): {num_peaks}")
print(f"基因座种类数: {num_markers}")
print(f"染料种类数: {num_dyes}")

# Descriptive stats for numeric columns
print("\n峰高 (Height) 和峰大小 (Size) 描述性统计:")
print(df_annex1_long[['Height', 'Size']].describe().to_markdown(numalign="left", stralign="left"))


# Value counts for categorical columns
print("\n基因座 (Marker) 频数分布:")
print(df_annex1_long['Marker'].value_counts().to_markdown(numalign="left", stralign="left"))

print("\n染料 (Dye) 频数分布:")
print(df_annex1_long['Dye'].value_counts().to_markdown(numalign="left", stralign="left"))

print("\n等位基因 (Allele) 频数分布 (前20个):")
print(df_annex1_long['Allele'].value_counts().head(20).to_markdown(numalign="left", stralign="left"))

# Frequency of 'OL' peaks (often considered artifacts/伪峰)
ol_peaks_count = df_annex1_long[df_annex1_long['Allele'] == 'OL'].shape[0]
print(f"\n'OL' 伪峰总数: {ol_peaks_count}")
if num_peaks > 0:
    print(f"'OL' 伪峰占总峰数的比例: {ol_peaks_count / num_peaks:.4f}")

# --- 4. Exploratory Analysis & Plots (Updated Terminology) ---
print("\n--- 探索性分析与图表 ---")

# Plot 1: Distribution of Height
plt.figure(figsize=(12, 6))
sns.histplot(df_annex1_long['Height'], bins=50, kde=True)
plt.title('所有样本峰高 (Height) 分布')
plt.xlabel('峰高')
plt.ylabel('频数')
plt.xlim(0, df_annex1_long['Height'].quantile(0.99)) # Limit x-axis for better visualization, excluding extreme outliers
plt.show()

# Plot 2: Distribution of Size
plt.figure(figsize=(12, 6))
sns.histplot(df_annex1_long['Size'], bins=50, kde=True)
plt.title('所有样本峰大小 (Size) 分布')
plt.xlabel('峰大小 (bp)') # Size is in base pairs (bp)
plt.ylabel('频数')
plt.xlim(0, df_annex1_long['Size'].max() * 1.05) # Show full range
plt.show()


# Plot 3: Peak counts per Marker (基因座)
plt.figure(figsize=(14, 7))
marker_counts = df_annex1_long['Marker'].value_counts().sort_values(ascending=False)
sns.barplot(x=marker_counts.index, y=marker_counts.values, palette='viridis')
plt.title('每个基因座 (Marker) 下的峰数量')
plt.xlabel('基因座 (Marker)')
plt.ylabel('峰数量')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 4: Peak counts per Dye
plt.figure(figsize=(8, 6))
dye_counts = df_annex1_long['Dye'].value_counts().sort_values(ascending=False)
sns.barplot(x=dye_counts.index, y=dye_counts.values, palette='plasma')
plt.title('每个染料 (Dye) 下的峰数量')
plt.xlabel('染料 (Dye)')
plt.ylabel('峰数量')
plt.show()


# Plot 5: Distribution of Height per Marker (Boxplot)
# Select top N markers for clarity if too many
top_markers = marker_counts.head(10).index.tolist()
df_top_markers = df_annex1_long[df_annex1_long['Marker'].isin(top_markers)]

plt.figure(figsize=(15, 8))
sns.boxplot(x='Marker', y='Height', data=df_top_markers, palette='tab10')
plt.title('不同基因座 (Marker) 下峰高 (Height) 分布')
plt.xlabel('基因座 (Marker)')
plt.ylabel('峰高')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df_top_markers['Height'].quantile(0.95)) # Limit y-axis for better visualization
plt.tight_layout()
plt.show()

# Plot 6: Frequency of 'OL' peaks per Marker
ol_per_marker = df_annex1_long[df_annex1_long['Allele'] == 'OL']['Marker'].value_counts()
if not ol_per_marker.empty:
    plt.figure(figsize=(14, 7))
    ol_per_marker = ol_per_marker.sort_values(ascending=False)
    sns.barplot(x=ol_per_marker.index, y=ol_per_marker.values, palette='Reds_r')
    plt.title('每个基因座 (Marker) 下的 \'OL\' 伪峰数量')
    plt.xlabel('基因座 (Marker)')
    plt.ylabel('\'OL\' 伪峰数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("\n没有检测到 'OL' 伪峰，跳过绘制 'OL' 伪峰数量图表。")


# --- 5. Sample-Level Aggregation and Analysis by NOC ---
print("\n--- 样本级别聚合特征及按贡献者人数 (NOC) 分析 ---")

# Aggregate basic features per sample, keeping NOC
sample_agg = df_annex1_long.groupby(['Sample File', 'NOC']).agg(
    total_peaks=('Allele', 'size'), # Total number of peaks
    avg_height=('Height', 'mean'),
    max_height=('Height', 'max'),
    avg_size=('Size', 'mean'),
    max_size=('Size', 'max'),
    ol_peak_count=('Allele', lambda x: (x == 'OL').sum()) # Count 'OL' peaks per sample
).reset_index()

# Drop samples where NOC extraction failed if any
initial_sample_agg_rows = sample_agg.shape[0]
sample_agg.dropna(subset=['NOC'], inplace=True)
sample_agg['NOC'] = sample_agg['NOC'].astype(int) # Convert NOC to integer
if sample_agg.shape[0] < initial_sample_agg_rows:
     print(f"注意: 移除了 {initial_sample_agg_rows - sample_agg.shape[0]} 个无法提取 NOC 的样本。")


print("\n每个样本文件的聚合特征 (包含NOC，前5行):")
print(sample_agg.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\n每个样本文件的聚合特征 描述性统计:")
print(sample_agg.describe().to_markdown(numalign="left", stralign="left"))

# Describe aggregated features grouped by NOC
if 'NOC' in sample_agg.columns and sample_agg['NOC'].nunique() > 1:
    print("\n按贡献者人数 (NOC) 分组的样本聚合特征描述性统计:")
    print(sample_agg.groupby('NOC')[['total_peaks', 'avg_height', 'max_height', 'ol_peak_count']].describe().to_markdown(numalign="left", stralign="left"))


    # Plot 7: Distribution of Total Peaks per Sample, colored by NOC
    plt.figure(figsize=(12, 6))
    sns.histplot(data=sample_agg, x='total_peaks', hue='NOC', bins=30, kde=True, multiple="stack")
    plt.title('不同贡献者人数 (NOC) 下的总峰数量分布')
    plt.xlabel('总峰数量')
    plt.ylabel('样本文件数量')
    plt.legend(title='NOC')
    plt.show()

    # Plot 8: Total Peaks vs NOC (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='NOC', y='total_peaks', data=sample_agg, palette='viridis')
    plt.title('不同贡献者人数 (NOC) 下的总峰数量分布')
    plt.xlabel('贡献者人数 (NOC)')
    plt.ylabel('总峰数量')
    plt.show()

    # Plot 9: Max Height vs NOC (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='NOC', y='max_height', data=sample_agg, palette='plasma')
    plt.title('不同贡献者人数 (NOC) 下的最大峰高分布')
    plt.xlabel('贡献者人数 (NOC)')
    plt.ylabel('最大峰高')
    plt.ylim(0, sample_agg['max_height'].quantile(0.95)) # Limit y-axis for better visualization
    plt.show()

    # Plot 10: OL Peak Count vs NOC (Boxplot)
    if sample_agg['ol_peak_count'].sum() > 0: # Only plot if there are OL peaks
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='NOC', y='ol_peak_count', data=sample_agg, palette='Reds_r')
        plt.title('不同贡献者人数 (NOC) 下的 \'OL\' 伪峰数量分布')
        plt.xlabel('贡献者人数 (NOC)')
        plt.ylabel('\'OL\' 伪峰数量')
        plt.show()
    else:
        print("\n所有样本中没有 'OL' 伪峰，跳过绘制按 NOC 分组的 'OL' 伪峰数量图表。")

else:
    print("\n样本中只检测到一种贡献者人数 (NOC)，跳过按 NOC 分组的分析和图表。")


print("\n描述性统计分析与探索性分析完成。")
