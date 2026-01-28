import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from dataset import SpeckleFlowDataset
from model import SpecklePINN

# ================= 配置 =================
CONFIG = {
    'roots': {
        'group_680W': '/data/zm/2026.1.12_testdata/1.15_150_680W/',
        'group_gaoyuzhi': '/data/zm/2026.1.12_testdata/gaoyuzhi/'
    },
    'window_size_us': 100000,
    'step_size_us': 50000,
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lambda_flow': 1.0,
    'lambda_fit': 10.0,
    'save_dir': '/data/zm/2026.1.12_testdata/1.26_PINN_result',

    # 🔥🔥🔥 严酷验证：保留流速列表 🔥🔥🔥
    # 训练集将看不到这些流速，必须靠物理规律“猜”出来
    'holdout_flows': [0.8, 1.8, 2.5]
}


def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # 1. 准备数据 (物理隔离)
    print("Loading TRAIN dataset...")
    # train 模式：排除 holdout_flows
    train_ds = SpeckleFlowDataset(CONFIG['roots'], mode='train',
                                  holdout_flows=CONFIG['holdout_flows'],
                                  window_size_us=CONFIG['window_size_us'],
                                  step_size_us=CONFIG['step_size_us'])

    print("Loading VAL dataset...")
    # val 模式：只包含 holdout_flows
    val_ds = SpeckleFlowDataset(CONFIG['roots'], mode='val',
                                holdout_flows=CONFIG['holdout_flows'],
                                window_size_us=CONFIG['window_size_us'],
                                step_size_us=CONFIG['step_size_us'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    print(f"Data split: Train={len(train_ds)} slices, Val={len(val_ds)} slices")

    # 2. 模型
    model = SpecklePINN().to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 3. 训练
    print("Start Training (Rigorous Physics Mode)...")
    history = {'train_loss': [], 'val_loss': []}

    # 定义 Fit Loss 的权重 (可选：给头部更高权重)
    # 既然归一化修好了，暂时用均匀权重
    # Fit loss 权重：强调早期下降段（你关心的前 1ms / 5ms）
    tau_us = (model.tau_grid.detach().cpu().numpy() * 1e6).astype(np.float32)
    w = np.ones_like(tau_us, dtype=np.float32)
    w[tau_us <= 1000.0] = 5.0
    w[(tau_us > 1000.0) & (tau_us <= 5000.0)] = 2.0
    # 归一化：让平均权重为 1，避免等效 lambda_fit 突变
    w = w / (np.mean(w) + 1e-9)
    fit_weights = torch.from_numpy(w).to(CONFIG['device'])

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", unit="batch")

        for batch in pbar:
            g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
            aux = batch['aux_input'].to(CONFIG['device']).float()
            v_label = batch['flow_label'].to(CONFIG['device']).float()
            m_val = batch['k_factor'].to(CONFIG['device']).float()

            optimizer.zero_grad()

            out = model(g2_obs, aux, m_val)

            # Loss 计算
            g2_hat = out['g2_hat']

            # Fit Loss
            loss_fit = torch.mean(fit_weights * (g2_hat - g2_obs) ** 2)

            # Flow Loss
            v_pred = out['v_pred']
            loss_flow = torch.mean((v_pred - v_label) ** 2)

            loss = CONFIG['lambda_fit'] * loss_fit + CONFIG['lambda_flow'] * loss_flow

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

            pbar.set_postfix({
                'L': f"{loss.item():.2f}",
                'Fit': f"{loss_fit.item():.2f}",
                'Flow': f"{loss_flow.item():.2f}"
            })

        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
        history['train_loss'].append(avg_loss)

        # === 验证 ===
        model.eval()
        val_loss_sum = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                g2_obs = batch['g2_curve'].to(CONFIG['device']).float()
                aux = batch['aux_input'].to(CONFIG['device']).float()
                v_label = batch['flow_label'].to(CONFIG['device']).float()
                m_val = batch['k_factor'].to(CONFIG['device']).float()

                out = model(g2_obs, aux, m_val)
                v_err = torch.abs(out['v_pred'] - v_label).mean()

                val_loss_sum += v_err.item()
                val_count += 1

        avg_val_mae = val_loss_sum / val_count if val_count > 0 else 0.0
        history['val_loss'].append(avg_val_mae)

        scheduler.step(avg_val_mae)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val MAE (Unseen Flows): {avg_val_mae:.4f}")

        if epoch > 0 and avg_val_mae < min(history['val_loss'][:-1]):
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val MAE (Holdout)')
    plt.legend()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'training_result.png'))
    print("Rigorous Training Complete.")


if __name__ == "__main__":
    main()