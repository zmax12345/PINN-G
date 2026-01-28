import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SpeckleFlowDataset
from model import SpecklePINN
import os

# ================= 配置 =================
CONFIG = {
    'roots': {
        'group_680W': '/data/zm/2026.1.12_testdata/1.15_150_680W/',
        'group_gaoyuzhi': '/data/zm/2026.1.12_testdata/gaoyuzhi/'
    },
    'window_size_us': 100000,
    'step_size_us': 50000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': '/data/zm/2026.1.12_testdata/1.26_PINN_result/best_model.pth',
    # 必须与 train.py 一致
    'holdout_flows': [0.8, 1.8, 2.5]
}


def evaluate_rigorous():
    print("Loading VAL dataset (Holdout Only)...")
    # 必须使用 mode='val' 且传入 holdout_flows
    val_ds = SpeckleFlowDataset(CONFIG['roots'], mode='val',
                                holdout_flows=CONFIG['holdout_flows'],
                                window_size_us=CONFIG['window_size_us'],
                                step_size_us=CONFIG['step_size_us'])

    # 不打乱，按顺序取，或者随机取
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    model = SpecklePINN().to(CONFIG['device'])
    if not os.path.exists(CONFIG['model_path']):
        print("Model not found!")
        return
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    # 准备存储结果，按流速分类
    results = {}  # {flow_label: {'pred': [], 'err': []}}

    print("Running Inference...")
    with torch.no_grad():
        for batch in val_loader:
            g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
            aux = batch['aux_input'].to(CONFIG['device']).float()
            v_label = batch['flow_label'].item()
            m_val = batch['k_factor'].to(CONFIG['device']).float()

            out = model(g2_obs, aux, m_val)
            v_pred = out['v_pred'].item()
            g2_hat = out['g2_hat'].cpu().numpy()[0]
            g2_obs = g2_obs.cpu().numpy()[0]

            if v_label not in results:
                results[v_label] = {'preds': [], 'errs': [], 'curves': []}

            results[v_label]['preds'].append(v_pred)
            results[v_label]['errs'].append(abs(v_pred - v_label))
            # 只存几个曲线画图用
            if len(results[v_label]['curves']) < 2:
                results[v_label]['curves'].append((g2_obs, g2_hat, v_pred))

    # --- 绘图与统计 ---
    unique_flows = sorted(results.keys())
    fig, axes = plt.subplots(len(unique_flows), 2, figsize=(12, 4 * len(unique_flows)))
    if len(unique_flows) == 1: axes = axes.reshape(1, -1)

    print("\n========= 严酷验证结果报告 =========")

    for i, flow in enumerate(unique_flows):
        data = results[flow]
        mean_mae = np.mean(data['errs'])
        mean_pred = np.mean(data['preds'])
        std_pred = np.std(data['preds'])

        print(f"流速: {flow:.2f} mm/s")
        print(f"   -> 平均预测: {mean_pred:.2f} ± {std_pred:.2f}")
        print(f"   -> MAE: {mean_mae:.4f}")
        print(f"   -> 相对误差: {(mean_mae / flow) * 100:.2f}%")

        # 画左图：误差分布散点
        ax_scatter = axes[i, 0]
        ax_scatter.hist(data['preds'], bins=20, alpha=0.7, color='skyblue', label='Preds')
        ax_scatter.axvline(flow, color='red', linestyle='--', linewidth=2, label='Ground Truth')
        ax_scatter.set_title(f"Label v={flow:.2f} | MAE={mean_mae:.2f}")
        ax_scatter.legend()

        # 画右图：曲线拟合情况 (抽样)
        ax_curve = axes[i, 1]
        if len(data['curves']) > 0:
            obs, hat, pred_v = data['curves'][0]
            ax_curve.plot(obs, 'b.', alpha=0.5, label='Observed')
            ax_curve.plot(hat, 'r-', linewidth=2, label=f'PINN (v={pred_v:.2f})')
            ax_curve.set_title(f"Curve Fitting (Sample)")
            ax_curve.legend()
            ax_curve.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/data/zm/2026.1.12_testdata/1.26_PINN_result/rigorous_evaluation.png')
    print("====================================")
    print("结果图已保存至 rigorous_evaluation.png")


if __name__ == "__main__":
    evaluate_rigorous()