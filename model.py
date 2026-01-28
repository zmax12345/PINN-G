import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Grid 转为 秒
TAU_LAGS_US = np.unique(np.concatenate([
    # 0 ~ 0.5 ms : 10 us step
    np.arange(0, 500, 10),
    # 0.5 ~ 5 ms : 100 us step
    np.arange(500, 5001, 100),
    # 5 ~ 100 ms : 1 ms step
    np.arange(5000, 100001, 1000),
])).astype(np.float32)
TAU_GRID_SECONDS = TAU_LAGS_US * 1e-6


class SpecklePINN(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=128):
        super().__init__()
        self.register_buffer('tau_grid', torch.tensor(TAU_GRID_SECONDS))

        if input_dim is None:
            input_dim = int(self.tau_grid.numel())

        self.backbone = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )

        self.head_tau = nn.Linear(hidden_dim // 2, 1)
        self.head_beta = nn.Linear(hidden_dim // 2, 1)
        self.head_alpha = nn.Linear(hidden_dim // 2, 1)

        # 初始化 bias
        nn.init.constant_(self.head_tau.bias, 0.0)

    def forward(self, g2_curve, aux_input, m_value):
        x = torch.cat([g2_curve, aux_input], dim=1)
        feat = self.backbone(x)

        # 1. 预测 Tau_c (秒) [0.001, 0.1]
        tau_c = torch.sigmoid(self.head_tau(feat)) * (0.1 - 0.001) + 0.001

        # 2. Beta & Alpha
        beta = torch.sigmoid(self.head_beta(feat))  # beta: 0~1 (反差)
        alpha = torch.sigmoid(self.head_alpha(feat)) * 1.5 + 0.5

        # --- Physics Decoder ---
        t = self.tau_grid.unsqueeze(0) + 1e-9

        term = t / tau_c
        exponent = -2.0 * (term ** alpha)
        exponent = torch.clamp(exponent, min=-20.0, max=0.0)

        # 🔥🔥🔥 恢复物理公式：1.0 + beta * exp(...) 🔥🔥🔥
        # 你的 Dataset 现在是归一化到 1 (基线)，所以这里必须加 1.0
        g2_hat = 1.0 + beta * torch.exp(exponent)

        # --- Flow Prediction ---
        # v = m / tau_c
        v_pred = m_value / tau_c

        return {
            'tau_c': tau_c,
            'beta': beta,
            'alpha': alpha,
            'g2_hat': g2_hat,
            'v_pred': v_pred
        }