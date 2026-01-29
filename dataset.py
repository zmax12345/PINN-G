import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import correlate

# 96点 Grid (保持不变)
TAU_LAGS = np.unique(np.concatenate([
    # 0 ~ 0.5 ms : 10 us step
    np.arange(0, 500, 10),
    # 0.5 ~ 5 ms : 100 us step
    np.arange(500, 5001, 100),
    # 5 ~ 100 ms : 1 ms step
    np.arange(5000, 100001, 1000),
])).astype(np.int64)


class SpeckleFlowDataset(Dataset):
    def __init__(self, data_roots, mode='train', holdout_flows=None, window_size_us=100000, step_size_us=50000):
        self.window_size_us = int(window_size_us)
        self.step_size_us = int(step_size_us)
        self.tau_lags = TAU_LAGS
        self.data_cache = []
        self.samples = []
        self.mode = mode
        self.holdout_flows = holdout_flows if holdout_flows is not None else []

        # 设定积分时间 (Integration Time) 用于将事件转为光强信号
        # 10us 一个 bin，足以分辨 5000us 的延迟
        self.dt_us = 10

        print(f"Dataset ({mode}) initializing with Signal Processing Engine...")
        self._load_all_files(data_roots)
        print(f"Dataset ({mode}) initialized: {len(self.samples)} samples.")

    def _load_all_files(self, roots):
        file_idx_counter = 0
        for group_name, root_dir in roots.items():
            if not os.path.exists(root_dir):
                print(f"Warning: Directory not found: {root_dir}")
                continue

            # 🔥🔥🔥 核心修改：明确指定每一组的 m 值 (mm) 🔥🔥🔥
            # m = 像素物理尺寸(mm) * 散斑像素大小(pixels)

            if 'gaoyuzhi' in group_name:
                # 第一组老数据
                current_m = 0.012915
                print(f"   -> Group '{group_name}': Matched 'gaoyuzhi', set m = {current_m:.6f} mm")

            elif 'group_680W' in group_name or '680W' in root_dir:
                # 第二组老数据
                current_m = 0.011167
                print(f"   -> Group '{group_name}': Matched '680W', set m = {current_m:.6f} mm")

            elif 'group_580W' in group_name:
                # 🔥 这里填你新数据的名字和算出来的 m 值
                # 例如：像素 0.00345mm * 散斑 3.2px = 0.01104
                current_m = 0.011808  # <--- 请修改这里！
                print(f"   -> Group '{group_name}': Matched 'new_experiment', set m = {current_m:.6f} mm")

            files = glob.glob(os.path.join(root_dir, "*.csv"))
            for fpath in files:
                try:
                    fname = os.path.basename(fpath)
                    try:
                        name_clean = fname.replace("_clip.csv", "").replace("mm.csv", "").replace("mm", "")
                        flow_val = float(name_clean)
                    except:
                        continue

                    # 严格划分
                    is_holdout = False
                    for hv in self.holdout_flows:
                        if abs(flow_val - hv) < 0.01:
                            is_holdout = True
                            break

                    if self.mode == 'train' and is_holdout: continue
                    if self.mode == 'val' and not is_holdout: continue

                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        df = pd.read_csv(f, header=None, usecols=[0, 1, 2], dtype=str, engine='c', on_bad_lines='skip')
                    df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(np.int64)
                    max_vals = df.max().values
                    tin_col_idx = np.argmax(max_vals)
                    tin_array = np.ascontiguousarray(df.iloc[:, tin_col_idx].sort_values().values)

                    if len(tin_array) < 1000: continue
                    duration = tin_array[-1] - tin_array[0]
                    if duration > 60 * 1e6 or duration <= 0: continue

                    self.data_cache.append(tin_array)
                    self._make_slices_fast(file_idx_counter, tin_array, flow_val, current_m)
                    file_idx_counter += 1

                except Exception as e:
                    print(f"Skip {fpath}: {e}")

    def _make_slices_fast(self, file_idx, t_all, label, m_val):
        t_min, t_max = t_all[0], t_all[-1]
        start_times = np.arange(t_min, t_max - self.window_size_us + 1, self.step_size_us)
        if len(start_times) == 0: return

        end_times = start_times + self.window_size_us
        idx_starts = np.searchsorted(t_all, start_times)
        idx_ends = np.searchsorted(t_all, end_times)
        counts = idx_ends - idx_starts

        valid_mask = counts > 1000  # 至少要有事件
        for i in np.where(valid_mask)[0]:
            self.samples.append((file_idx, int(idx_starts[i]), int(idx_ends[i]), np.float32(label), np.float32(m_val)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start, end, label, m_val = self.samples[idx]
        ts = self.data_cache[file_idx][start:end]
        ts = ts - ts[0]  # 归零

        # === 🔥 核心重构：基于 FFT 的标准光强自相关 ===
        # 1. 转为光强信号 (Intensity Trace)
        # 窗口总时长 window_size_us，分辨率 dt_us
        num_bins = self.window_size_us // self.dt_us
        # 使用直方图统计每个 dt 内的事件数 -> I(t)
        I_t, _ = np.histogram(ts, bins=num_bins, range=(0, self.window_size_us))
        I_t = I_t.astype(np.float32)

        # 2. 计算自相关 G2(\tau) = <I(t)I(t+\tau)>
        # 使用 FFT 加速卷积：Correlate I_t with itself
        # mode='full' 返回长度 2*N-1，中心是 0 滞后
        acf = correlate(I_t, I_t, mode='full')

        # 取右半部分 (正滞后)
        center = len(acf) // 2
        acf_right = acf[center:]  # 长度 num_bins

        # 3. 归一化：g2 = <I(t)I(t+\tau)> / <I(t)>^2
        # 注意：correlate 是求和不是求平均，所以要除以重叠的 bin 数量
        normalization_array = np.arange(num_bins, 0, -1).astype(np.float32)
        G2 = acf_right / (normalization_array + 1e-9)  # G2(\tau) raw

        mean_I = np.mean(I_t)
        baseline = mean_I ** 2

        if baseline > 1e-9:
            g2_final = G2 / baseline
        else:
            g2_final = np.ones_like(G2)

        # 4. 映射到我们的 TAU_LAGS 网格
        # TAU_LAGS 单位是 us，我们的 dt_us 是 10us
        # 所以 index = tau / 10
        indices = (self.tau_lags // self.dt_us).astype(np.int64)
        indices = np.clip(indices, 0, len(g2_final) - 1)

        g2_feature = g2_final[indices]

        # 5. 简单清洗
        g2_feature = np.nan_to_num(g2_feature, nan=1.0)
        # 物理上 g2 通常从 >1 开始衰减到 1。
        # 如果噪声导致 < 0.5，视为异常
        g2_feature = np.maximum(g2_feature, 0.5)

        # Aux input (Mean Intensity)
        log_intensity = np.log10(mean_I + 1e-6).astype(np.float32)

        return {
            'g2_curve': torch.from_numpy(g2_feature),
            'aux_input': torch.tensor([log_intensity]),
            'flow_label': torch.tensor([label]),
            'k_factor': torch.tensor([m_val])
        }