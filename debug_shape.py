import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import glob
import os

# ================= 配置 =================
CONFIG = {
    # 请填入你的真实路径
    'empty_file': '/data/zm/2026.1.12_testdata/noblood/239.csv',  # 空管
    'blood_file': '/data/zm/2026.1.12_testdata/gaoyuzhi/1.5mm_clip.csv',  # 典型的血液数据
    'window_us': 100000,  # 100ms
    'dt_us': 10,
    'max_lag_us': 10000,  # 看前 10ms
    'num_windows': 10  # 取 10 个窗口做统计
}


def load_It(file_path):
    try:
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2], dtype=str, engine='c', on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(np.int64)
        df = df[df.iloc[:, 1] <= 768]  # ROI
        tin = df.iloc[:, 2].sort_values().values
        t_start = tin[0]
        duration = tin[-1] - t_start
        num_bins = int(duration // CONFIG['dt_us']) + 1
        I_t, _ = np.histogram(tin - t_start, bins=num_bins, range=(0, num_bins * CONFIG['dt_us']))
        return I_t.astype(np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calc_g2_slice(I_slice):
    mean_I = np.mean(I_slice)
    if mean_I < 1e-3: return None

    acf = correlate(I_slice, I_slice, mode='full', method='fft')
    center = len(acf) // 2
    acf_right = acf[center:]
    norm = np.arange(len(I_slice), 0, -1)
    g2 = (acf_right / (norm + 1e-9)) / (mean_I ** 2 + 1e-9)
    return g2[:CONFIG['max_lag_us'] // CONFIG['dt_us']]


def analyze_stability(I_t, label):
    win_bins = CONFIG['window_us'] // CONFIG['dt_us']
    step = len(I_t) // (CONFIG['num_windows'] + 1)

    g2_list = []
    stats = []

    for i in range(CONFIG['num_windows']):
        start = (i + 1) * step
        sl = I_t[start: start + win_bins]
        if np.sum(sl) < 100: continue

        g2 = calc_g2_slice(sl)
        if g2 is not None:
            g2_list.append(g2)
            stats.append({
                'mean_I': np.mean(sl),
                'count': np.sum(sl),
                'var': np.var(sl)
            })

    if not g2_list: return None, None, None

    g2_arr = np.array(g2_list)
    g2_mean = np.mean(g2_arr, axis=0)
    g2_std = np.std(g2_arr, axis=0)

    # 打印统计信息
    avg_count = np.mean([s['count'] for s in stats])
    print(f"\n📊 [{label}] Statistics (over {len(stats)} windows):")
    print(f"   Avg Event Count: {avg_count:.1f}")
    print(f"   Avg Intensity: {np.mean([s['mean_I'] for s in stats]):.4f}")

    return g2_mean, g2_std, g2_arr


def main():
    print("Loading data...")
    I_empty = load_It(CONFIG['empty_file'])
    I_blood = load_It(CONFIG['blood_file'])

    if I_empty is None or I_blood is None: return

    # 1. 稳定性分析
    mean_noise, std_noise, all_noise = analyze_stability(I_empty, "Empty Tube")
    mean_blood, std_blood, _ = analyze_stability(I_blood, "Blood 1.2mm")

    t = np.arange(len(mean_noise)) * CONFIG['dt_us'] / 1000.0

    plt.figure(figsize=(15, 10))

    # 子图1: 空管稳定性
    plt.subplot(2, 2, 1)
    for i in range(len(all_noise)):
        plt.plot(t, all_noise[i], 'r-', alpha=0.1)  # 画出所有单次采样
    plt.plot(t, mean_noise, 'r-', linewidth=2, label='Mean Noise')
    plt.fill_between(t, mean_noise - std_noise, mean_noise + std_noise, color='r', alpha=0.2)
    plt.title(f"Stability Check: Empty Tube ({CONFIG['num_windows']} windows)")
    plt.xlabel("Lag (ms)")
    plt.ylabel("g2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 血液 vs 噪声
    plt.subplot(2, 2, 2)
    plt.plot(t, mean_blood, 'b-', label='Blood 1.2mm')
    plt.plot(t, mean_noise, 'r--', label='Noise Template')
    plt.title("Signal vs Noise Template")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 尝试背景减除 (Alpha=1.0)
    plt.subplot(2, 1, 2)

    # 简单的直接减法： (g2_blood - 1) - (g2_noise - 1)
    # 这假设了加性噪声模型
    g2_corrected = (mean_blood - 1) - (mean_noise - 1) + 1

    plt.plot(t, mean_blood, 'b--', alpha=0.5, label='Original Blood')
    plt.plot(t, g2_corrected, 'g-', linewidth=2, label='Corrected (Blood - Noise)')
    plt.axhline(1.0, color='k', linestyle=':')
    plt.title("Hypothesis Test: Background Subtraction (Alpha=1.0)")
    plt.xlabel("Lag (ms)")
    plt.ylabel("g2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/data/zm/2026.1.12_testdata/1.26_PINN_result/2.4/rigorous_diagnosis.png')
    print("\n✅ Diagnosis saved to rigorous_diagnosis.png")
    print("Look at Subplot 3 (Green Line):")
    print("Does it look smoother and more monotonic than the Blue Line?")


if __name__ == "__main__":
    main()