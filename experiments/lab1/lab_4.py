import pandas as pd
import re
import numpy as np

# 请替换为您的实际文件路径
excel_file_path = r"..\..\data\data1_diff_nums.xlsx"

# --- 定义一些预处理参数 ---
# 报告阈值 (Reporting Threshold)，低于此高度的峰可能被视为噪声并过滤 (AMEL通常有自己的阈值或处理方式)
# 这里的阈值需要根据实际实验情况和数据质量进行调整
REPORTING_THRESHOLD = 5 # 示例阈值，表示50 RFU (Relative Fluorescent Units)

# 用于识别潜在伪峰的参数
# 常见STR位点的重复单位长度（bp）。实际应用中需要精确查找每个Marker的重复单位。
# 这里列出常见核心位点的重复单位，AMEL是性别位点，处理方式不同
REPEAT_UNITS = {
    'D8S1179': 4, 'D21S11': 4, 'D7S820': 4, 'CSF1PO': 4, 'D3S1358': 4,
    'TH01': 4, 'D13S317': 4, 'D16S539': 4, 'D2S1338': 4, 'D19S433': 4,
    'vWA': 4, 'TPOX': 4, 'D18S51': 4, 'D5S818': 4, 'FGA': 4,
    # AMEL是性别位点，通常不需要考虑stutter/+A，其大小是固定的X(约105bp)和Y(约111bp)
    'AMEL': None
    # 您数据中可能还有其他Marker，需要补充其重复单位
}

# 伪峰高度比阈值
STUTTER_RATIO_THRESHOLD = 0.15 # 口吃峰通常小于主峰的15% (取决于位点和大小)
PLUSA_RATIO_THRESHOLD = 0.10 # 加A峰通常小于其对应的非加A峰的10% (取决于位点和大小)
SIZE_TOLERANCE = 0.2 # 允许的大小偏差 (bp)

# --- 1. 数据载入与初步整理 ---
print(f"正在从文件载入数据: {excel_file_path}")
try:
    df_wide = pd.read_excel(excel_file_path)
    print("数据载入成功。")
    print(f"原始数据形状: {df_wide.shape}")
    # print("原始数据前5行:")
    # print(df_wide.head())

except FileNotFoundError:
    print(f"错误: 文件未找到，请检查路径是否正确: {excel_file_path}")
    exit()
except Exception as e:
    print(f"载入文件时发生错误: {e}")
    exit()

# 确定需要融化的列组
allele_cols = [col for col in df_wide.columns if col.startswith('Allele ')]
size_cols = [col for col in df_wide.columns if col.startswith('Size ')]
height_cols = [col for col in df_wide.columns if col.startswith('Height ')]

# 识别固定列
id_cols = ['Sample File', 'Marker', 'Dye']

# 验证列数量是否一致 (Allele, Size, Height 应该是成对的)
if not (len(allele_cols) == len(size_cols) == len(height_cols)):
    print("警告: Allele, Size, Height 列数量不匹配，请检查原始数据格式。")

# 构建所有需要融化的列名列表
value_cols = allele_cols + size_cols + height_cols

# 执行融化操作 (将宽格式转为长格式)
# melt之后会有 Sample File, Marker, Dye, variable (e.g., 'Allele 1'), value
print("正在将数据从宽格式转换为长格式...")
df_long_melted = pd.melt(df_wide,
                         id_vars=id_cols,
                         value_vars=value_cols,
                         var_name='Variable',
                         value_name='Value')

# 从 'Variable' 列中提取 'Metric' (Allele, Size, Height) 和 'Peak_Index' (1, 2, ...)
# 使用正则表达式提取数字部分作为 Peak_Index，前面的文本作为 Metric
df_long_melted['Metric'] = df_long_melted['Variable'].apply(lambda x: x.split(' ')[0])
df_long_melted['Peak_Index'] = df_long_melted['Variable'].apply(lambda x: int(x.split(' ')[1]))

# 丢弃原始的 'Variable' 列
df_long_melted = df_long_melted.drop(columns=['Variable'])

# 使用 pivot_table 将 Metric 列重新转为列 (Allele, Size, Height)
# 索引是 Sample File, Marker, Dye, Peak_Index
print("正在重新组织峰信息...")
df_long = df_long_melted.pivot_table(index=id_cols + ['Peak_Index'],
                                     columns='Metric',
                                     values='Value',
                                     aggfunc='first') # aggfunc='first' 因为每个组合只有一个值

# 重置索引，将索引列转为普通列
df_long = df_long.reset_index()

# 确保列名规范化 (Pivot_table 可能会导致列名变成索引对象，这里强制转换为字符串)
df_long.columns.name = None # 移除 columns.name
df_long.columns = [str(col) for col in df_long.columns]

# 移除 Peak_Index 全为空的行 (这些是原始数据中对应峰位没有数据的)
# 注意：melt后如果某个(Allele, Size, Height)三元组有一个非空，就会保留。
# 在pivot_table后，如果原始三元组中有任何一个值，对应的行就会在df_long出现，但其他Metric可能为NaN
# 所以这里检查Allele, Size, Height是否都为NaN，如果是，说明原始该峰位无数据，可以丢弃。
# 实际上，如果原始数据是规范的，只要Allele/Size/Height有一个非空，其他两个通常也非空，
# 但为了安全，我们检查关键列（例如 Size 和 Height）是否为NaN
initial_rows = df_long.shape[0]
df_long.dropna(subset=['Size', 'Height'], inplace=True)
rows_after_dropna = df_long.shape[0]
if initial_rows > rows_after_dropna:
     print(f"丢弃了 {initial_rows - rows_after_dropna} 行因 Size 或 Height 缺失的峰。")

print(f"转换为长格式后的数据形状: {df_long.shape}")
print("长格式数据前5行:")
print(df_long.head())

# --- 从 Sample File 文件名中提取关键信息 ---
# 假设文件名格式类似: XXX_RD14-YYYY-ID1_ID2..._Conditions_Suffix.fsa
# NOC (Number of Contributors) 可以通过计算 ID1, ID2... 的数量来确定
def parse_filename(filename):
    try:
        # 查找 RD14-YYYY- 后面的部分
        match = re.search(r'RD14-\d{4}-(\d[0-9_]+)-\d', filename) # 匹配 ID1_ID2... 后面的 -数字
        if match:
            contributor_part = match.group(1) # 提取 ID1_ID2... 部分
            sample_ids = contributor_part.split('_')
            noc = len(sample_ids)
            return noc, sample_ids
        else:
            # 如果格式不完全匹配，尝试其他方法或返回默认值/None
             # 备用逻辑：查找文件名中连续的数字和下划线组成的段，排除前面的编号和后面的条件
            parts = filename.replace('.fsa', '').split('_')
            # 寻找看起来像 ID 列表的部分 (例如，可能是第三或第四段)
            # 这是一个启发式方法，可能需要根据实际文件名结构调整
            for part in parts:
                if re.fullmatch(r'\d+(_\d+)+', part): # 匹配 数字_数字_... 模式
                     sample_ids = part.split('_')
                     return len(sample_ids), sample_ids
                elif re.fullmatch(r'\d+', part) and len(parts) > 2 and parts[2] == part: # 如果是单个数字且在第三段
                     # 可能是单个人样本的文件名格式不同，这里按1人处理
                     # 但附件1是混合样本，所以这个分支可能不常用
                     # 更可能是 NOC 判断逻辑的问题，先返回None
                     pass # 继续寻找更符合混合样本的模式

            # 如果所有尝试都失败，返回 None
            print(f"警告: 无法解析文件名 '{filename}' 的贡献者信息。")
            return None, None

    except Exception as e:
        print(f"解析文件名 '{filename}' 时发生错误: {e}")
        return None, None

# 对每个样本文件（Sample File），只需要解析一次文件名。
# 提取唯一的 Sample File 列表进行解析，然后将结果合并回去
unique_sample_files = df_long['Sample File'].unique()
sample_info_list = []
for filename in unique_sample_files:
    noc, sample_ids = parse_filename(filename)
    sample_info_list.append({'Sample File': filename, 'NOC': noc, 'Sample_IDs': sample_ids})

sample_info_df = pd.DataFrame(sample_info_list)

# 将解析出的信息合并回主数据框
df_long = pd.merge(df_long, sample_info_df, on='Sample File', how='left')

# 检查是否有文件名解析失败的样本
if df_long['NOC'].isnull().any():
    failed_files = df_long[df_long['NOC'].isnull()]['Sample File'].unique()
    print(f"警告: 部分样本文件名解析失败，NOC 信息缺失: {list(failed_files)}")

print("\n数据载入、初步整理及文件名解析完成。")
print("带有解析信息的长格式数据前5行:")
print(df_long.head())


# --- 2. 数据清洗与质量控制 ---

print("\n--- 数据清洗与质量控制 ---")

# 过滤低于报告阈值 (RT) 的峰
# 通常对所有Marker应用RT，但AMEL有时例外。这里我们对所有非OL的峰应用RT。
# OL峰的处理更复杂，可能需要单独考虑其高度和性质。
# 对于问题一（人数判断），可能需要保留所有高于RT的峰，包括OL。
# 对于问题二/三（比例/基因型），低OL峰通常需要过滤。
# 这里的处理是过滤所有低于RT的非OL峰。OL峰保留，但其Allele标记'OL'。
# 也可以选择先不过滤，只标记，让下游模型决定如何处理。
# 我们先进行RT过滤（非AMEL，非OL），并标记OL。
initial_peaks_count = df_long.shape[0]

# 标记 'OL' 峰
df_long['Is_OL'] = df_long['Allele'] == 'OL'
print(f"标记了 {df_long['Is_OL'].sum()} 个 'OL' 峰。")

# 过滤低于 RT 的非 AMEL 峰 (AMEL通常按存在性判断，且阈值可能不同)
# 假设AMEL不需要按RT过滤，或者有单独的阈值。这里简化处理：非AMEL且Height < RT的峰被过滤。
# 如果AMEL有自己的RT，需要修改这里的条件。
filtered_rt_count = df_long[
    (~df_long['Marker'].isin(REPEAT_UNITS) | (df_long['Marker'].isin(REPEAT_UNITS) & df_long['Marker'].apply(lambda x: REPEAT_UNITS[x] is not None))) & # 不是AMEL或是非AMEL
    (df_long['Height'] < REPORTING_THRESHOLD)
].shape[0]

df_long_cleaned = df_long[
    (df_long['Marker'].isin(REPEAT_UNITS) & (df_long['Marker'].apply(lambda x: REPEAT_UNITS[x] is None) | (df_long['Height'] >= REPORTING_THRESHOLD))) | # 是AMEL 或 (非AMEL 且 Height >= RT)
    (~df_long['Marker'].isin(REPEAT_UNITS)) # 不是已知Marker的，先保留
].copy() # 使用.copy()避免SettingWithCopyWarning

print(f"过滤了 {filtered_rt_count} 个低于报告阈值 ({REPORTING_THRESHOLD} RFU) 的非AMEL峰。")
print(f"清洗后剩余峰数量: {df_long_cleaned.shape[0]}")

# 潜在伪峰识别 (Stutter 和 +A) - 标记而非移除
# 这个过程需要按样本和基因座分组，并比较峰之间的 Size 和 Height 关系。
# 由于比较是成对的，相对复杂，且AMEL位点不适用此逻辑。
# 我们只对已知重复单位的非AMEL位点进行标记。
print("正在识别潜在的 Stutter 和 +A 伪峰...")

df_long_cleaned['Is_Stutter_Candidate'] = False
df_long_cleaned['Is_PlusA_Candidate'] = False

# 过滤出需要进行伪峰分析的行 (非AMEL且已知重复单位)
df_for_artifact = df_long_cleaned[
    df_long_cleaned['Marker'].isin(REPEAT_UNITS) &
    df_long_cleaned['Marker'].apply(lambda x: REPEAT_UNITS[x] is not None)
].copy() # 复制以避免SettingWithCopyWarning

if not df_for_artifact.empty:
    # 按样本、基因座分组并按大小排序
    df_for_artifact = df_for_artifact.sort_values(by=['Sample File', 'Marker', 'Size'])

    # 在每个组内，使用 shift() 与相邻峰进行比较
    df_for_artifact['Prev_Size'] = df_for_artifact.groupby(['Sample File', 'Marker'])['Size'].shift(1)
    df_for_artifact['Prev_Height'] = df_for_artifact.groupby(['Sample File', 'Marker'])['Height'].shift(1)
    df_for_artifact['Next_Size'] = df_for_artifact.groupby(['Sample File', 'Marker'])['Size'].shift(-1)
    df_for_artifact['Next_Height'] = df_for_artifact.groupby(['Sample File', 'Marker'])['Height'].shift(-1)

    # 定义一个辅助函数来获取Marker的重复单位
    def get_repeat_unit(marker):
        return REPEAT_UNITS.get(marker)

    # 识别 Stutter (主要 N-1 stutter)
    # 峰 Size 大约比前一个峰大一个重复单位，且当前峰比前一个峰矮（前一个峰是主峰），且满足高度比阈值
    # 这里的逻辑是：当前峰 (p1) 是否是 前一个峰 (p_prev) 的 stutter 峰
    # 即 p1.Size ~ p_prev.Size + repeat_unit, p1.Height < p_prev.Height * stutter_ratio
    # 然而，根据数据结构，前一个峰是 df_for_artifact.shift(1)。所以我们应该判断 shift(1) 峰是否是 当前峰的 stutter
    # 也就是：当前峰 (p1), 前一个峰 (p_prev = df_for_artifact.shift(1))
    # 如果 p1.Size ~ p_prev.Size + repeat_unit (即 p_prev.Size ~ p1.Size - repeat_unit)
    # 并且 p_prev.Height < p1.Height * stutter_ratio_threshold
    # 那么 p_prev 是 p1 的 stutter 峰。我们在 p_prev 那一行标记 Is_Stutter_Candidate 为 True。
    # 由于我们在当前行操作，我们标记的是 *当前行* 的峰是否是 *前一个峰* 的 stutter
    # 或者，我们标记的是 *当前峰* 是否是 *下一个峰* 的 stutter (N-1 stutter出现在主峰前)
    # 让我们标记当前峰是否是 *后面更高峰* 的 N-1 stutter 候选
    # 简化逻辑：只看紧邻的前一个峰，如果当前峰大小约为 前一个峰 + repeat_unit 且当前峰高远大于前一个峰高
    # 那么前一个峰可能是当前峰的 stutter。
    # 实际操作：当前行是 Peak(Size, Height)。判断它是否是 Peak(Prev_Size, Prev_Height) 的 stutter。
    # 即 Prev_Size 约等于 Size - repeat_unit 且 Prev_Height < Height * ratio
    stutter_condition = (
        (df_for_artifact['Marker'].apply(get_repeat_unit).notnull()) & # Marker有重复单位
        (np.abs(df_for_artifact['Size'] - df_for_artifact['Prev_Size'] - df_for_artifact['Marker'].apply(get_repeat_unit)) < SIZE_TOLERANCE) & # Size difference is approx repeat_unit
        (df_for_artifact['Prev_Height'].notnull()) & # Previous peak exists
        (df_for_artifact['Height'] > df_for_artifact['Prev_Height']) & # Current peak is taller than previous
        (df_for_artifact['Prev_Height'] / df_for_artifact['Height'] < STUTTER_RATIO_THRESHOLD) # Previous peak height is within stutter ratio
    )
    # 在符合条件的行的 'Prev_Index' 对应的原始行标记 Is_Stutter_Candidate
    # 这需要在原始 df_long_cleaned 上进行，通过 Sample File, Marker, Peak_Index 来匹配
    # 找到符合条件的行的索引
    stutter_rows_to_flag = df_for_artifact[stutter_condition].copy()
    if not stutter_rows_to_flag.empty:
        # 确定哪些原始 Peak_Index 应该被标记
        # Stutter峰是Prev_Peak，所以我们需要前一行的Peak_Index
        stutter_rows_to_flag['Stutter_Peak_Index'] = df_for_artifact[stutter_condition].groupby(['Sample File', 'Marker'])['Peak_Index'].shift(1)
        # Merge this info back to df_long_cleaned to set the flag
        # Need to handle the case where the first peak in a group is flagged - shift(1) will be NaN
        # Let's simplify the flagging: flag the *current* peak if its *next* neighbor is a potential main peak it stutters from.
        # Simplified Stutter Flagging: Current peak is stutter if Size ~ Next_Size - repeat_unit and Height < Next_Height * ratio
        stutter_condition_simplified = (
            (df_for_artifact['Marker'].apply(get_repeat_unit).notnull()) &
            (np.abs(df_for_artifact['Size'] - (df_for_artifact['Next_Size'] - df_for_artifact['Marker'].apply(get_repeat_unit))) < SIZE_TOLERANCE) &
            (df_for_artifact['Next_Height'].notnull()) &
            (df_for_artifact['Height'] < df_for_artifact['Next_Height']) & # Stutter peak should be shorter than main peak
            (df_for_artifact['Height'] / df_for_artifact['Next_Height'] < STUTTER_RATIO_THRESHOLD)
        )
        df_for_artifact.loc[stutter_condition_simplified, 'Is_Stutter_Candidate'] = True
        print(f"标记了 {stutter_condition_simplified.sum()} 个潜在的 Stutter 伪峰。")
    else:
         print("未识别到符合条件的 Stutter 伪峰。")


    # 识别 +A (N+1 加A)
    # 峰 Size 约等于前一个峰 Size + 1 bp，且前一个峰是其对应的非加A峰，且满足高度比阈值
    # 实际操作：当前行是 Peak(Size, Height)。判断它是否是 Peak(Prev_Size, Prev_Height) 的 +A。
    # 即 Size 约等于 Prev_Size + 1 且 Height < Prev_Height * ratio
    # 或者更常见的判断：当前峰是 +A 峰，它紧邻的前一个峰是对应的非加A峰 (N峰)
    # Simplified +A Flagging: Current peak is +A if Size ~ Prev_Size + 1 and Height < Prev_Height * ratio
    plus_a_condition_simplified = (
        (df_for_artifact['Marker'].apply(get_repeat_unit).notnull()) & # 仅对非AMEL位点
        (np.abs(df_for_artifact['Size'] - df_for_artifact['Prev_Size'] - 1) < SIZE_TOLERANCE) & # Size difference is approx 1 bp
        (df_for_artifact['Prev_Height'].notnull()) & # Previous peak exists
        (df_for_artifact['Height'] < df_for_artifact['Prev_Height']) & # +A peak is typically shorter than main peak
        (df_for_artifact['Height'] / df_for_artifact['Prev_Height'] < PLUSA_RATIO_THRESHOLD) # +A peak height is within ratio
    )
    df_for_artifact.loc[plus_a_condition_simplified, 'Is_PlusA_Candidate'] = True
    print(f"标记了 {plus_a_condition_simplified.sum()} 个潜在的 +A 伪峰。")
else:
    print("没有非AMEL且已知重复单位的Marker数据，跳过伪峰识别。")


# 将伪峰标记合并回原始清洗后的数据框
# 需要确保df_long_cleaned的索引和df_for_artifact的索引一致或者能通过关键列合并
# 由于df_for_artifact是df_long_cleaned的子集且经过了sort，直接更新可能不对齐
# 最佳方式是通过共同的标识列合并
# 标识列是 Sample File, Marker, Dye, Peak_Index
# 创建一个包含标识列和新标记列的DataFrame
artifact_flags = df_for_artifact[id_cols + ['Peak_Index', 'Is_Stutter_Candidate', 'Is_PlusA_Candidate']]

# 合并回主数据框 (只更新非AMEL部分的标记，因为AMEL的标记列是False)
# 确保合并键是正确的
df_long_cleaned = pd.merge(df_long_cleaned, artifact_flags, on=id_cols + ['Peak_Index'], how='left', suffixes=('', '_new'))
# 更新标记列，将新合并的标记填充到原列中（对于AMEL等没有新标记的，保持False）
df_long_cleaned['Is_Stutter_Candidate'] = df_long_cleaned['Is_Stutter_Candidate_new'].fillna(df_long_cleaned['Is_Stutter_Candidate']).astype(bool)
df_long_cleaned['Is_PlusA_Candidate'] = df_long_cleaned['Is_PlusA_Candidate_new'].fillna(df_long_cleaned['Is_PlusA_Candidate']).astype(bool)
# 移除临时列
df_long_cleaned = df_long_cleaned.drop(columns=['Is_Stutter_Candidate_new', 'Is_PlusA_Candidate_new'])


# 重新检查清洗后的数据结构和前几行
print("\n完成数据清洗与质量控制。")
print(f"最终处理后的数据形状: {df_long_cleaned.shape}")
print("最终处理后的数据前5行:")
print(df_long_cleaned.head())

# 打印一些汇总信息
print("\n--- 清洗后数据汇总信息 ---")
print(f"样本数量: {df_long_cleaned['Sample File'].nunique()}")
print(f"基因座数量: {df_long_cleaned['Marker'].nunique()}")
print(f"每个样本的峰数量分布:\n{df_long_cleaned.groupby('Sample File').size().describe()}")
print(f"每个基因座的峰数量分布:\n{df_long_cleaned.groupby('Marker').size().describe()}")
print(f"包含'OL'峰的基因座:\n{df_long_cleaned[df_long_cleaned['Is_OL']]['Marker'].value_counts()}")
print(f"标记为潜在 Stutter 的峰数量: {df_long_cleaned['Is_Stutter_Candidate'].sum()}")
print(f"标记为潜在 +A 的峰数量: {df_long_cleaned['Is_PlusA_Candidate'].sum()}")

# 您现在可以使用 df_long_cleaned 进行后续的问题分析。
# 例如，对于问题一，您可以基于这个数据框提取特征；
# 对于问题二/三，您可以使用这个包含清洗和标记信息的峰列表。
# 确保已经有了处理后的 df_long_cleaned DataFrame
# ... (前面的数据预处理代码) ...

# 指定保存路径
output_excel_file_path = r"..\..\data\processed_data1.xlsx"

print(f"\n正在将处理后的数据保存到 Excel 文件: {output_excel_file_path}")

try:
    # index=False 表示不将DataFrame的索引保存为Excel的一列
    df_long_cleaned.to_excel(output_excel_file_path, index=False)
    print("数据成功保存到 Excel 文件。")
except Exception as e:
    print(f"保存 Excel 文件时发生错误: {e}")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # 确保 pandas 已导入
# --- 添加以下代码解决中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持中文的字体，如 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号'-'显示为方块的问题

# 假设 df_long 是您在第一步“数据载入与初步整理”后，但未进行清洗（特别是RT过滤）的数据框
# 假设 df_long_cleaned 是您在第二步“数据清洗与质量控制”后，应用了RT过滤等步骤的数据框
# 如果您直接运行了上面的代码，那么 df_long_cleaned 应该已经包含了RT过滤的结果。
# df_long 是 pivot_table 之后，但未dropna或过滤RT之前的版本。
# 为了进行对比，我们需要保留一个RT过滤之前的版本。
# 在之前的代码中，df_long 是 pivot_table 之后并进行了 initial dropna 的版本。
# df_long_cleaned 是在 df_long 的基础上进行了 RT 过滤等操作。
# 所以，我们将 df_long 作为“处理前”（至少是主要清洗前）的数据，df_long_cleaned 作为“处理后”的数据。

# --- 可视化对比处理前后数据 ---

print("\n--- 可视化对比处理前后数据 ---")

# 确保 Height 列是数值类型
df_long['Height'] = pd.to_numeric(df_long['Height'], errors='coerce')
df_long_cleaned['Height'] = pd.to_numeric(df_long_cleaned['Height'], errors='coerce')

# 移除 Height 为 NaN 的行，以避免绘图问题
df_long_viz = df_long.dropna(subset=['Height']).copy()
df_long_cleaned_viz = df_long_cleaned.dropna(subset=['Height']).copy()

# 定义一系列高度阈值用于对比
# 从0到一个合理的上限（例如，最高峰高或某个高百分位数）
max_height = max(df_long_viz['Height'].max(), df_long_cleaned_viz['Height'].max())
# 生成100个阈值点
thresholds = np.linspace(0, max_height, 200)

# 计算在每个阈值下，高于或等于该阈值的峰的数量
peaks_above_threshold_original = []
peaks_above_threshold_cleaned = []

print("正在计算不同高度阈值下的峰数量...")
for threshold in thresholds:
    count_original = df_long_viz[df_long_viz['Height'] >= threshold].shape[0]
    count_cleaned = df_long_cleaned_viz[df_long_cleaned_viz['Height'] >= threshold].shape[0]
    peaks_above_threshold_original.append(count_original)
    peaks_above_threshold_cleaned.append(count_cleaned)

print("计算完成，正在生成折线图...")

# 绘制折线图
plt.figure(figsize=(12, 7))

plt.plot(thresholds, peaks_above_threshold_original, label='处理前 (Original)', color='blue')
plt.plot(thresholds, peaks_above_threshold_cleaned, label=f'处理后 (RT >= {REPORTING_THRESHOLD} RFU)', color='red')

# 添加图表元素
plt.xlabel('峰高阈值 (RFU)')
plt.ylabel('高于或等于该阈值的峰数量')
plt.title('处理前后峰高分布对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(bottom=0) # 确保Y轴从0开始
plt.xlim(left=0)  # 确保X轴从0开始

# 显示图表
plt.show()

print("折线图已生成。")

# 还可以添加其他对比，例如：
# 1. 每个样本的峰数量对比 (柱状图)
# 2. 每个Marker的峰数量对比 (柱状图)

print("\n--- 更多对比 (柱状图) ---")

# 1. 每个样本的峰数量对比
peaks_per_sample_original = df_long_viz.groupby('Sample File').size()
peaks_per_sample_cleaned = df_long_cleaned_viz.groupby('Sample File').size()

# 为了对比，合并两个系列的样本列表，确保所有样本都显示
all_samples = pd.Index(peaks_per_sample_original.index.tolist() + peaks_per_sample_cleaned.index.tolist()).unique()
peaks_per_sample_original = peaks_per_sample_original.reindex(all_samples, fill_value=0)
peaks_per_sample_cleaned = peaks_per_sample_cleaned.reindex(all_samples, fill_value=0)

x = np.arange(len(all_samples)) # 样本数量
width = 0.35 # 柱状图宽度

plt.figure(figsize=(15, 7))
plt.bar(x - width/2, peaks_per_sample_original.values, width, label='处理前')
plt.bar(x + width/2, peaks_per_sample_cleaned.values, width, label='处理后')

plt.ylabel('峰数量')
plt.title('处理前后各样本峰数量对比')
plt.xticks(x, all_samples, rotation=90) # 设置样本文件名为X轴标签
plt.legend()
plt.tight_layout() # 调整布局以避免标签重叠
plt.show()

print("各样本峰数量对比图已生成。")

# 2. 每个Marker的峰数量对比
peaks_per_marker_original = df_long_viz.groupby('Marker').size()
peaks_per_marker_cleaned = df_long_cleaned_viz.groupby('Marker').size()

# 为了对比，合并两个系列的Marker列表，确保所有Marker都显示
all_markers = pd.Index(peaks_per_marker_original.index.tolist() + peaks_per_marker_cleaned.index.tolist()).unique()
peaks_per_marker_original = peaks_per_marker_original.reindex(all_markers, fill_value=0)
peaks_per_marker_cleaned = peaks_per_marker_cleaned.reindex(all_markers, fill_value=0)


x = np.arange(len(all_markers)) # Marker数量
width = 0.35 # 柱状图宽度

plt.figure(figsize=(15, 7))
plt.bar(x - width/2, peaks_per_marker_original.values, width, label='处理前')
plt.bar(x + width/2, peaks_per_marker_cleaned.values, width, label='处理后')

plt.ylabel('峰数量')
plt.title('处理前后各Marker峰数量对比')
plt.xticks(x, all_markers, rotation=90) # 设置Marker名称为X轴标签
plt.legend()
plt.tight_layout() # 调整布局以避免标签重叠
plt.show()

print("各Marker峰数量对比图已生成。")


