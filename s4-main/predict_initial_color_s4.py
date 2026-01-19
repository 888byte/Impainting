import numpy as np
import torch
import torch.nn as nn

from models.s4.s4d import S4D  # 跟训练一致


class S4InitialColorModel(nn.Module):
    def __init__(self, d_in=8, d_model=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([S4D(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, x):
        # x: [B, K, 8]
        h = self.in_proj(x)        # [B, K, d_model]
        h = h.transpose(1, 2)      # [B, d_model, K]  (S4D expects B,H,L)

        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out  # 兼容 (y, state)

        h = h.transpose(1, 2)      # [B, K, d_model]
        h = self.norm(h)
        h_last = h[:, -1, :]
        return self.head(h_last)   # [B,3]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="s4_initial_color.pt")
    ap.add_argument("--data", default="data/color_series.npy")
    ap.add_argument("--series_id", type=int, default=0, help="0..8 对应 NO1..NO9")
    ap.add_argument("--use_last_k", type=int, default=None, help="覆盖 ckpt 里的 K（可选）")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    K = int(ckpt.get("K", 1))
    if args.use_last_k is not None:
        K = args.use_last_k

    mean = ckpt["mean"]  # (8,)
    std = ckpt["std"]    # (8,)

    raw = np.load(args.data).astype(np.float32)  # [9,T,8]
    seq = raw[args.series_id]                    # [T,8]

    # 输入：最后 K 个点（现在附近）
    x_now = seq[-K:]                             # [K,8]
    x_now = np.where(np.isfinite(x_now), x_now, mean[None, :]).astype(np.float32)

    x_now_n = (x_now - mean) / std
    x_now_n = np.nan_to_num(x_now_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = S4InitialColorModel(d_in=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_now_n[None, :, :]).to(device)).cpu().numpy()[0]  # (3,)

    # 反归一化回 RGB
    pred_rgb = pred_n * std[:3] + mean[:3]
    true_rgb = seq[0, :3]

    print(f"NO{args.series_id+1}:")
    print("Predicted initial RGB:", pred_rgb)
    print("True initial RGB     :", true_rgb)


if __name__ == "__main__":
    main()
