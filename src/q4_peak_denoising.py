import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

GENE_ORDER = [
    "D8S1179", "D21S11", "D7S820", "CSF1PO",
    "D3S1358", "TH01", "D13S317", "D16S539",
    "D2S1338", "D19S433", "vWA", "TPOX",
    "D18S51", "AMEL", "D5S818", "FGA"
]


def load_data(file_path):
    data = pd.read_excel(file_path)
    assert len(data) % 16 == 0, "数据必须包含16的倍数行"
    total_samples = len(data) // 16
    print(f"成功加载{total_samples}组STR混合图谱")
    return data


def process_sample(sample_data,
                   min_peak_height=10,
                   shadow_threshold=0.05,
                   height_threshold_ratio=0.5):
    profile = []
    gene_loci = []
    for idx, row in sample_data.iterrows():
        gene = row.get('Marker', row.get('基因座', GENE_ORDER[idx % 16]))
        for col in row.index:
            if 'Height' in col or '高度' in col:
                val = row[col]
                if not np.isnan(val) and val > 0:
                    profile.append(val)
                    gene_loci.append(gene)

    profile = np.array(profile)
    if len(profile) == 0:
        print("警告：样本无原始数据")
        return None, None  # 返回None和空DataFrame

    # 基线校正和处理逻辑（同上）
    baseline = savgol_filter(profile, window_length=32, polyorder=3)
    corrected = profile - baseline
    corrected[corrected < 0] = 0
    filtered = savgol_filter(corrected, window_length=5, polyorder=2)

    unique_genes = list(set(gene_loci))
    gene_heights = {gene: [] for gene in unique_genes}
    for val, gene in zip(filtered, gene_loci):
        gene_heights[gene].append(val)

    gene_thresholds = {
        gene: np.mean(heights) * height_threshold_ratio
        for gene, heights in gene_heights.items() if len(heights) > 0
    }

    enhanced = filtered.copy()
    for i, (val, gene) in enumerate(zip(enhanced, gene_loci)):
        threshold = gene_thresholds.get(gene, 0)
        if val < threshold:
            enhanced[i] = 0

    peaks, props = find_peaks(
        enhanced,
        height=min_peak_height,
        prominence=5,
        distance=15
    )

    if len(peaks) > 0:
        peak_heights = enhanced[peaks]
        main_peaks = [peaks[i] for i in np.where(peak_heights > shadow_threshold * np.max(peak_heights))[0]]
    else:
        main_peaks = []

    # 生成峰信息DataFrame
    peak_info = []
    for pos in main_peaks:
        if pos < len(gene_loci) and pos < len(enhanced):
            gene = gene_loci[pos]
            height = enhanced[pos]
            peak_info.append({
                '样本ID': sample_data.iloc[0].get('样本ID', i // 16 + 1),  # 提取样本ID
                '基因座': gene,
                '位置': pos,
                '高度': height
            })
    peak_df = pd.DataFrame(peak_info)

    return {
        '原始信号': profile,
        '降噪信号': filtered,
        '真实峰位置': main_peaks,
        '增强信号': enhanced,
        '基因座': gene_loci
    }, peak_df


def visualize_result(result, sample_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(result['原始信号'], label='原始信号', alpha=0.5)
    plt.title(f'样本{sample_id} - 原始图谱')
    plt.subplot(2, 1, 2)
    plt.plot(result['增强信号'], label='处理后信号（低峰置零）')
    plt.scatter(result['真实峰位置'], result['增强信号'][result['真实峰位置']], c='red', s=200)
    plt.title(f'样本{sample_id} - 保留{len(result["真实峰位置"])}个有效峰')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'sample_{sample_id}.png'))


def main_denoising_workflow(input_file='Data_T1.xlsx', output_dir='峰值阈值处理结果'):
    data = load_data(input_file)
    os.makedirs(output_dir, exist_ok=True)

    all_peak_data = []  # 存储所有样本的峰信息
    sample_id_counter = 1

    for i in range(0, len(data), 16):
        sample_data = data.iloc[i:i + 16]
        print(f"\n处理样本{sample_id_counter}...")

        result, peak_df = process_sample(sample_data, height_threshold_ratio=0.5)
        if result is None:
            sample_id_counter += 1
            continue

        # 保存单个样本峰信息到Excel
        sample_output_file = os.path.join(output_dir, f'sample_{sample_id_counter}_peaks.xlsx')
        with pd.ExcelWriter(sample_output_file) as writer:
            peak_df.to_excel(writer, sheet_name='峰信息', index=False)
            # 可追加其他数据（如原始信号、降噪信号）
            pd.DataFrame(result['增强信号'], columns=['增强后高度']).to_excel(writer, sheet_name='增强信号',
                                                                              index=False)

        visualize_result(result, sample_id_counter, output_dir)
        print(f"检测到{len(result['真实峰位置'])}个有效峰，结果保存于{sample_output_file}")

        all_peak_data.append(peak_df)
        sample_id_counter += 1

    # 汇总所有样本峰信息到一个Excel
    summary_df = pd.concat(all_peak_data, ignore_index=True)
    summary_path = os.path.join(output_dir, 'all_samples_peaks_summary.xlsx')
    summary_df.to_excel(summary_path, index=False)
    print(f"\n所有样本峰信息已汇总至{summary_path}")

    print("处理完成！")


if __name__ == "__main__":
    main_denoising_workflow()