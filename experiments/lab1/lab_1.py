"""

"""
# 导入必要的库
import pandas as pd          # 用于数据处理和分析，特别是DataFrame操作
import matplotlib.pyplot as plt # 用于创建静态、交互式和动画可视化
import os                    # 用于操作系统相关的任务，例如创建目录
import re                    # 用于正则表达式操作，这里用来提取列名中的数字

# --- 文件路径配置 ---
# 定义输入Excel文件的完整路径。请将此路径修改为您实际的文件位置。
data_path = r"..\..\data\data1_diff_nums.xlsx"
# 定义保存输出图表的目录路径。请将此路径修改为您希望保存图片的文件夹位置。
output_dir = r"..\..\lab1"

# 确保输出目录存在。如果目录不存在，os.makedirs会自动创建。
os.makedirs(output_dir, exist_ok=True)

# --- 数据加载与重塑函数 ---
def load_and_restructure_str_data(file_path):
    """
    读取STR Excel数据（原始格式，每个样本一行，多个Allele/Size/Height列），
    并将其重塑为长格式DataFrame，其中每行代表一个单独的峰。

    Args:
        file_path (str): 输入Excel文件的完整路径。

    Returns:
        pandas.DataFrame: 重塑后的长格式DataFrame，包含 'Sample File',
                          'Marker', 'Dye', 'Allele', 'Size', 'Height' 列。
                          如果读取失败或数据为空，返回空的DataFrame。
    """
    print(f"正在读取数据文件: {file_path}")
    try:
        # 读取Excel文件到DataFrame (宽格式)
        df_wide = pd.read_excel(file_path)
        print("成功读取数据。原始数据字段示例 (前2行):\n", df_wide.head(2))

    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径是否正确: {file_path}")
        return pd.DataFrame() # 返回一个空的DataFrame表示失败
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return pd.DataFrame() # 返回一个空的DataFrame表示失败

    # 识别 Allele, Size, Height 列组，并动态确定最大等位基因编号
    # 查找所有以 'Allele' 开头且后面跟着数字的列名
    allele_cols_base = [col for col in df_wide.columns if re.match(r'Allele \d+', col)]
    # 从这些列名中提取数字，找到最大值
    allele_numbers = [int(re.search(r'\d+', col).group()) for col in allele_cols_base]
    # 确定最大等位基因编号。如果没有任何匹配的列，默认为0。
    max_allele = max(allele_numbers) if allele_numbers else 0
    print(f"根据列名检测到最大等位基因编号: {max_allele}")

    # 存储重塑后的峰记录列表
    peak_records = []

    # 定义在重塑过程中保持不变的列
    id_vars = ['Sample File', 'Marker', 'Dye']

    # 检查必需的固定列是否存在于原始DataFrame中
    if not all(col in df_wide.columns for col in id_vars):
        print(f"错误: 必需的列 {id_vars} 不存在于Excel文件中。请检查列名。")
        return pd.DataFrame() # 返回一个空的DataFrame表示失败

    # 遍历原始DataFrame的每一行（每个样本的记录）
    for index, row in df_wide.iterrows():
        # 获取当前行的固定信息
        sample_file = row['Sample File']
        marker = row['Marker']
        dye = row['Dye']

        # 遍历每个可能的等位基因位点（从1到最大编号）
        for i in range(1, max_allele + 1):
            # 构建当前等位基因、大小和高度的列名
            allele_col = f'Allele {i}'
            size_col = f'Size {i}'
            height_col = f'Height {i}'

            # 检查这些列在当前行是否存在
            if allele_col in row and size_col in row and height_col in row:
                # 获取当前峰的等位基因、大小和高度值
                allele_val = row[allele_col]
                size_val = row[size_col]
                height_val = row[height_col]

                # 尝试将 Size 和 Height 转换为数字。如果遇到非数字值(如'OL')，
                # errors='coerce' 会将其转换为 NaN。
                numeric_size = pd.to_numeric(size_val, errors='coerce')
                numeric_height = pd.to_numeric(height_val, errors='coerce')

                # 只有当 Size 和 Height 都成功转换为非NaN的数字时，才将其视为有效峰并添加记录
                if pd.notna(numeric_size) and pd.notna(numeric_height):
                    peak_records.append({
                        'Sample File': sample_file, # 保留原始文件名
                        'Marker': marker,         # 保留基因座名称
                        'Dye': dye,               # 保留染料信息
                        'Allele': allele_val,     # 保留原始等位基因值 (可能包含'OL')
                        'Size': numeric_size,     # 片段大小 (bp)
                        'Height': numeric_height  # 峰高 (RFU)
                    })
                # else:
                    # 可以选择打印警告，如果 Size 或 Height 预期是数字但不是
                    # if pd.isna(numeric_size) or pd.isna(numeric_height):
                    #     print(f"警告: 跳过非数字峰数据 - Sample File: {sample_file}, Marker: {marker}, Allele {i}: Size='{size_val}', Height='{height_val}'")

    # 将收集到的峰记录列表转换为长格式DataFrame
    df_long = pd.DataFrame(peak_records)

    # 如果DataFrame非空，打印重塑后的数据示例
    if not df_long.empty:
        print("\n数据重塑完成。长格式数据字段示例 (前5行):\n", df_long.head())
        print(f"重塑后总共检测到 {len(df_long)} 个有效峰记录。")
    else:
         print("\n数据重塑完成，但未找到任何有效峰记录。请检查输入文件内容。")


    return df_long

# --- 可视化函数 ---
def plot_locus_scatter(df_long, marker_name):
    """
    根据指定基因座，绘制所有样本在该基因座上有效峰的 Size vs Height 散点图。

    Args:
        df_long (pandas.DataFrame): 包含STR峰数据的长格式DataFrame。
        marker_name (str): 要绘制的基因座名称。
    """
    print(f"\n正在绘制基因座: {marker_name} 的散点图...")

    # 筛选出属于当前基因座的数据
    marker_data = df_long[df_long['Marker'] == marker_name].copy() # 使用 .copy() 避免SettingWithCopyWarning

    # 如果筛选后数据为空，打印信息并返回
    if marker_data.empty:
        print(f"在数据中未找到基因座: {marker_name} 的有效峰数据，跳过绘制。")
        return # 结束函数执行

    # 定义颜色映射：将染料通道（如B, G, Y, R）映射到Matplotlib颜色名称
    color_map = {'B': 'blue', 'G': 'green', 'Y': 'yellow', 'R': 'red'}

    # 根据Dye列的值为每个点分配颜色。使用 .fillna('gray') 为不在映射中的染料设置默认颜色。
    colors = marker_data['Dye'].map(color_map).fillna('gray')

    # 创建图表和轴对象
    plt.figure(figsize=(12, 7)) # 设置图表尺寸

    # 绘制散点图：
    # x轴为Size，y轴为Height
    # c参数根据colors列表为每个点设置颜色
    # s参数设置点的大小
    # alpha设置点的透明度
    # edgecolors和linewidth设置点边缘的颜色和宽度
    plt.scatter(marker_data['Size'], marker_data['Height'],
                c=colors,
                s=60, alpha=0.8, edgecolors='w', linewidth=0.5)

    # --- 可选：标注每个点的Allele信息 ---
    # 注意：在数据点很多时，文本标注会使图表非常混乱，通常不建议默认开启。
    # 如果您需要标注，请取消下面代码块的注释，并根据需要调整fontsize。
    # for i, row in marker_data.iterrows():
    #     # 标注文本格式： Allele值 @ Size bp\nHeight RFU
    #     annotation_text = f"{row['Allele']} @ {row['Size']:.1f}bp\n{row['Height']}RFU"
    #     plt.text(row['Size'], row['Height'],
    #              annotation_text,
    #              ha='center', va='bottom', # 水平居中，垂直靠下
    #              fontsize=7, color='black') # 字体大小和颜色

    # 设置图表标题和坐标轴标签
    plt.xlabel("Size (bp)", fontsize=12)      # x轴标签：片段大小 (碱基对)
    plt.ylabel("Height (RFU)", fontsize=12)   # y轴标签：峰高 (相对荧光单位)
    plt.title(f"STR Peaks for Locus: {marker_name}", fontsize=15) # 图表标题，包含基因座名称

    # 添加图例：显示不同染料通道对应的颜色
    # 创建一个Line2D对象列表，用于在图例中显示圆点和颜色
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Dye {dye_code} ({color.capitalize()})',
                                  markerfacecolor=color, markersize=10, alpha=0.8)
                       for dye_code, color in color_map.items()]
    # 添加图例到图表
    plt.legend(handles=legend_elements, title="Dye Channels", loc='best') # loc='best' 尝试找到最佳位置

    # 添加网格线，提高可读性
    plt.grid(ls="--", alpha=0.4) # ls="--" 设置为虚线，alpha设置透明度

    # 自动调整图表布局，防止元素（如标签、图例）重叠
    plt.tight_layout()

    # 保存图表到文件
    output_filename = f"{marker_name}_peaks_scatter.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath, dpi=300) # dpi=300 设置分辨率，保证图片清晰度
    print(f"图表已保存至: {output_filepath}")

    # 显示图表
    plt.show()

# --- 主执行块 ---
if __name__ == "__main__":
    # 调用函数加载并重塑数据
    df_processed = load_and_restructure_str_data(data_path)

    # 检查处理后的数据是否为空
    if not df_processed.empty:
        # 获取数据中所有唯一的基因座名称
        available_markers = df_processed['Marker'].unique()
        print("\n数据中可用的基因座:", available_markers)

        # 定义您希望绘制的基因座列表。您可以根据 available_markers 修改此列表。
        markers_to_plot = ["D8S1179", "D21S11"] # 示例：绘制D8S1179和D21S11

        # 遍历要绘制的基因座列表，并调用绘图函数
        for marker in markers_to_plot:
             plot_locus_scatter(df_processed, marker)

        # 示例：如果您想绘制所有基因座，可以取消下面两行的注释
        # print("\n正在绘制所有基因座的图表...")
        # for marker in available_markers:
        #     plot_locus_scatter(df_processed, marker)

    else:
        print("未加载到有效数据，无法生成图表。请检查文件路径和内容格式。")

