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
    np.arange(5000, 150001, 1000),
    # 🔥🔥🔥 新增：100ms ~ 400ms : 5 ms step 🔥🔥🔥
    # 对于慢速流，不需要太密，5ms 一个点足够了
    #np.arange(100000, 200001, 5000),
])).astype(np.int64)


class SpeckleFlowDataset(Dataset):
    def __init__(self, data_roots, mode='train', holdout_flows=None, window_size_us=150000, step_size_us=50000):
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

        print(f"Dataset ({mode}) initializing with ROI Filter (Col <= 768)...")
        self._load_all_files(data_roots)
        print(f"Dataset ({mode}) initialized: {len(self.samples)} samples.")

    def _load_all_files(self, roots):
        file_idx_counter = 0
        for group_name, root_dir in roots.items():
            if not os.path.exists(root_dir):
                print(f"Warning: Directory not found: {root_dir}")
                continue

            # --- 1. 匹配物理参数 m ---
            if 'gaoyuzhi' in group_name:
                current_m = 0.014611
                print(f"   -> Group '{group_name}': Matched 'gaoyuzhi', set m = {current_m:.6f} mm")
            elif 'group_680W' in group_name or '680W' in root_dir:
                current_m = 0.0105
                print(f"   -> Group '{group_name}': Matched '680W', set m = {current_m:.6f} mm")
            elif 'group_580' in group_name:
                # 新数据
                current_m = 0.0114853
                print(f"   -> Group '{group_name}': Matched 'group_580W', set m = {current_m:.6f} mm")
            elif 'group_122' in group_name:
                # 新数据
                current_m = 0.010154
                print(f"   -> Group '{group_name}': Matched 'group_122', set m = {current_m:.6f} mm")
            elif 'group_pianzhen1' in group_name:
                # 新数据
                current_m = 0.010157
                print(f"   -> Group '{group_name}': Matched 'group_pianzhen1', set m = {current_m:.6f} mm")
            elif 'group_2.3' in group_name:
                # 新数据
                current_m = 0.010099
                print(f"   -> Group '{group_name}': Matched 'group_2.3', set m = {current_m:.6f} mm")
            else:
                # 默认值 (以防万一)
                current_m = 0.011167
                print(f"   -> Group '{group_name}': Unknown, using default m = {current_m:.6f} mm")

            # --- 2. 读取文件 ---
            files = glob.glob(os.path.join(root_dir, "*.csv"))
            for fpath in files:
                try:
                    fname = os.path.basename(fpath)
                    try:
                        name_clean = fname.replace("_clip.csv", "").replace("mm.csv", "").replace("mm", "")
                        flow_val = float(name_clean)
                    except:
                        continue

                    # 严格划分 Holdout
                    is_holdout = False
                    for hv in self.holdout_flows:
                        if abs(flow_val - hv) < 0.01:
                            is_holdout = True
                            break

                    if self.mode == 'train' and is_holdout: continue
                    if self.mode == 'val' and not is_holdout: continue

                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        # 读取前3列：Row(0), Col(1), Tin(2)
                        df = pd.read_csv(f, header=None, usecols=[0, 1, 2], dtype=str, engine='c', on_bad_lines='skip')

                    df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(np.int64)

                    # 🔥🔥🔥 核心修改：ROI 过滤 🔥🔥🔥
                    # 既然你知道：Col 0=Row, Col 1=Col(x), Col 2=Tin
                    # 我们只保留 Col(x) <= 768 的数据

                    original_count = len(df)
                    df = df[df.iloc[:, 1] <= 768]  # 过滤掉 x > 768 的噪声区域

                    # 如果过滤后没数据了，跳过
                    if len(df) < 1000:
                        continue

                    # 提取时间列 (Col 2)
                    tin_array = np.ascontiguousarray(df.iloc[:, 2].sort_values().values)

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

        # 过滤掉事件过少的窗口 (因为ROI裁剪后事件数会减少，这里可以适当降低阈值，或者保持1000)
        valid_mask = counts > 1000
        for i in np.where(valid_mask)[0]:
            self.samples.append((file_idx, int(idx_starts[i]), int(idx_ends[i]), np.float32(label), np.float32(m_val)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start, end, label, m_val = self.samples[idx]
        ts = self.data_cache[file_idx][start:end]
        ts = ts - ts[0]  # 归零

        # === 基于 FFT 的标准光强自相关 ===
        num_bins = self.window_size_us // self.dt_us
        I_t, _ = np.histogram(ts, bins=num_bins, range=(0, self.window_size_us))
        I_t = I_t.astype(np.float32)



        acf = correlate(I_t, I_t, mode='full')
        center = len(acf) // 2
        acf_right = acf[center:]

        normalization_array = np.arange(num_bins, 0, -1).astype(np.float32)
        G2 = acf_right / (normalization_array + 1e-9)

        mean_I = np.mean(I_t)
        baseline = mean_I ** 2

        if baseline > 1e-9:
            g2_final = G2 / baseline
        else:
            g2_final = np.ones_like(G2)

        indices = (self.tau_lags // self.dt_us).astype(np.int64)
        indices = np.clip(indices, 0, len(g2_final) - 1)
        g2_feature = g2_final[indices]

        # 5. 简单清洗
        g2_feature = np.nan_to_num(g2_feature, nan=1.0)

        g2_feature = np.maximum(g2_feature, 0.5)

        # 注意：这里归一化后，g2 变成了 1.0 -> 0.x 的形状
        # 而不是物理上的 1+beta -> 1.0
        # 这需要 model.py 配合修改公式！

        log_intensity = np.log10(mean_I + 1e-6).astype(np.float32)

        return {
            'g2_curve': torch.from_numpy(g2_feature),
            'aux_input': torch.tensor([log_intensity]),
            'flow_label': torch.tensor([label]),
            'k_factor': torch.tensor([m_val])
        }
