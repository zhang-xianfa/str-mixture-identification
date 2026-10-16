import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 文件路径设置 ---
excel_file_path = r"..\..\data\data1_diff_nums.xlsx"

# --- 2. 读取Excel文件 ---
try:
    # 尝试读取所有 sheet
    df_raw = pd.read_excel(excel_file_path, sheet_name=None)
    # 合并所有 sheet 的数据
    df_raw = pd.concat(df_raw.values(), ignore_index=True)

except FileNotFoundError:
    print(f"Error: File not found at {excel_file_path}")
    exit() # 如果文件不存在，退出程序
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

# --- 3. 数据处理: 从宽格式转换为长格式 ---

# 假设我们只处理第一个样本文件的数据
sample_files = df_raw['Sample File'].unique()
if len(sample_files) == 0:
    print("Error: No sample data found in the Excel file.")
    exit()

# 选择第一个样本文件的数据
first_sample_file = sample_files[0]
df_sample = df_raw[df_raw['Sample File'] == first_sample_file].copy()

# 创建一个空的DataFrame来存储所有峰的“长格式”数据
peaks_list = []

# 获取所有可能的 Allele/Size/Height 列名
allele_cols = [f'Allele {i}' for i in range(1, 101)]
size_cols = [f'Size {i}' for i in range(1, 101)]
height_cols = [f'Height {i}' for i in range(1, 101)]

# 确保这些列存在于DataFrame中
valid_allele_cols = [col for col in allele_cols if col in df_sample.columns]
valid_size_cols = [col for col in size_cols if col in df_sample.columns]
valid_height_cols = [col for col in height_cols if col in df_sample.columns]

# 找到共同存在的列索引
num_peaks_per_locus = min(len(valid_allele_cols), len(valid_size_cols), len(valid_height_cols))

if num_peaks_per_locus == 0:
    print("Error: Could not find valid 'Allele', 'Size', and 'Height' columns in the expected format.")
    exit()

# 遍历每个基因座（行）
for index, row in df_sample.iterrows():
    marker = row['Marker']
    dye = row['Dye'] # 获取染料信息，可能用于颜色映射

    # 遍历所有可能的峰列组合
    for i in range(num_peaks_per_locus):
        allele_col = valid_allele_cols[i]
        size_col = valid_size_cols[i]
        height_col = valid_height_cols[i]

        # 检查 Size 和 Height 列是否存在且有有效值
        # 使用 pd.notna() 更稳妥地检查 NaN
        if pd.notna(row[size_col]) and pd.notna(row[height_col]):
             # 将 Size 和 Height 转换为数值类型，处理可能的错误
            try:
                size = float(row[size_col])
                height = float(row[height_col])
                allele = row[allele_col] # 可以是数值、字符串（如OL, X, Y）或 NaN

                # 只添加 Height 大于0的峰
                if height > 0:
                    peaks_list.append({
                        'Marker': marker,
                        'Dye': dye,
                        'Size': size,
                        'Height': height,
                        'Allele': allele if pd.notna(allele) else None # 将NaN转换为None
                    })
            except (ValueError, TypeError):
                # Ignore rows where Size/Height cannot be converted to numeric
                continue


# 将列表转换为DataFrame
peaks_df = pd.DataFrame(peaks_list)

if peaks_df.empty:
    print(f"No valid peak data found for plotting in sample file: {first_sample_file}")
    exit()

# --- 4. 定义基因座对应的颜色和图例标签 ---
# 根据Excel中的Dye列和原始图中的颜色分布来定义颜色映射
# 这是一个基于观察和常见STR panel的颜色映射示例
# 你可以根据实际需求调整这些颜色
locus_colors_map = {
    # 根据您提供的原始图片和常见染料通道大致对应
    'D8S1179': 'blue',       # Blue
    'D21S11': 'darkorange',  # Blue? - Using Orange based on original image feel
    'D7S820': 'forestgreen', # Blue? - Using Green based on original image feel
    'CSF1PO': 'red',         # Blue? - Using Red based on original image feel
    'D3S1358': 'darkgreen',  # Green
    'TH01': 'purple',        # Green
    'D13S317': 'olivedrab',  # Green? - Using Olive
    'D16S539': 'c',          # Green? - Using Cyan
    'D2S1338': 'darkslategray', # Green? - Using Dark Slate Gray
    'D19S433': 'darkviolet', # Yellow? - Using Dark Violet
    'vWA': 'gold',           # Yellow
    'TPOX': 'sienna',        # Yellow? - Using Sienna (Brownish)
    'D18S51': 'gray',        # Yellow? - Using Gray
    'AMEL': 'teal',          # Red
    'D5S818': 'hotpink',     # Red? - Using Hot Pink
    'FGA': 'deepskyblue',    # Red? - Using Deep Sky Blue
    'DYS391': 'magenta',     # Red? - Using Magenta - Note: DYS391 not in sample data, but in original legend.
    'TPX': 'orange',         # Red? - Using Orange - Note: TPX not in sample data, but in original legend.
}

# 创建图例标签列表 (只包含实际数据中存在的Markers)
unique_markers_in_data = peaks_df['Marker'].unique()

# 尝试按照一个常见的顺序排序Marker，并为每个Marker分配颜色和图例句柄
# 您可能需要根据您的实际Panel和期望的图例顺序调整此列表
ordered_markers = [
    'D8S1179', 'D21S11', 'D7S820', 'CSF1PO',
    'TH01', 'TPOX', 'D5S818', 'D18S51',
    'D13S317', 'D16S539', 'vWA', 'TPX', # TPX included for potential completeness, but check data
    'D3S1358', 'AMEL', 'DYS391', 'FGA', # DYS391 included, but check data
    'D2S1338', 'D19S433' # Add others from your data if needed
]

# Filter ordered_markers to only include those present in the data
markers_to_plot = [m for m in ordered_markers if m in unique_markers_in_data]

# Create legend handles for markers present in data, maintaining order
legend_handles = []
legend_labels = []
for marker in markers_to_plot:
    color = locus_colors_map.get(marker, 'black') # Get color, default to black
    # Create a placeholder line plot for the legend entry
    # Use linestyle='-' to match the plot type, and marker='o' if you use markers on the plot
    handle = plt.Line2D([], [], color=color, marker='o', linestyle='-', markersize=5, label=marker)
    legend_handles.append(handle)
    legend_labels.append(marker) # Use marker name as label


# --- 5. 绘制图表 ---

plt.figure(figsize=(12, 7)) # 调整图表大小

# 绘制线图，按基因座分组并分配颜色
labeled_peaks_info = [] # To store info for labeling

for marker in markers_to_plot:
    locus_data = peaks_df[peaks_df['Marker'] == marker].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Sort data by Size before plotting lines
    locus_data = locus_data.sort_values(by='Size')

    color = locus_colors_map.get(marker, 'black') # Get color for this marker

    # Plot all points for the current marker using plot (line + markers)
    # Use marker='o' to keep the circles visible, and linestyle='-' to connect them
    plt.plot(locus_data['Size'], locus_data['Height'], color=color, marker='o', linestyle='-', linewidth=1, markersize=5)

    # Collect points for labeling (using the sorted data is fine)
    for index, row in locus_data.iterrows():
        # Label conditions: Allele is not 'OL' and Height is above threshold
        if pd.notna(row['Allele']) and str(row['Allele']).upper() != 'OL':
             # Check if height is above a reasonable threshold for labeling
             # Adjust threshold as needed based on your data distribution
            if row['Height'] > 50: # Increased example threshold for labeling based on data scale
                 labeled_peaks_info.append((row['Size'], row['Height'], str(row['Allele']), color))


# Add labels after plotting all points, using collected info
# Adjust the offset values (e.g., +0.5 for Size, +5 for Height) as needed
size_offset = 0.5
height_offset = 5

for (size, height, allele, color) in labeled_peaks_info:
     plt.text(size + size_offset, height + height_offset, allele,
             fontsize=8, ha='left', va='bottom', # Label position
             color=color) # Optional: color label by locus color

# --- 6. 设置图表属性 ---
plt.title(f'        STR Allele Size vs Peak Height for Sample: {first_sample_file} (Line Plot)') # Updated title
plt.xlabel('Size (bp)')
plt.ylabel('Height (RFU)')

# 设置 X 轴范围和刻度 (根据原始图片调整)
# 自动确定合理的X轴范围
x_min = peaks_df['Size'].min() - 5 if peaks_df['Size'].min() > 5 else 100
x_max = peaks_df['Size'].max() + 5 if peaks_df['Size'].max() < 525 else 530
plt.xlim(x_min, x_max)
plt.xticks(np.arange(int(x_min/10)*10, int(x_max/10)*10 + 11, 10)) # 每隔 10 bp 设置一个主刻度


# 设置 Y 轴范围和刻度 (根据数据最大值调整)
y_max = peaks_df['Height'].max() * 1.2 # Add 20% padding above max peak
y_max = max(y_max, 200) # Ensure a minimum height for clarity
plt.ylim(0, y_max)
plt.yticks(np.arange(0, y_max + 1, 50)) # 每隔 50 RFU 设置一个刻度


# 添加网格线
plt.grid(True, linestyle='-', alpha=0.6)

# 添加图例
# 使用之前收集的句柄和图例标签
plt.legend(handles=legend_handles, labels=legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)

# 调整布局以防止图例被裁剪
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整主图区域，为图例腾出空间

# 显示图表
plt.show()
