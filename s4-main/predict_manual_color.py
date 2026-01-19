import numpy as np
import torch
import torch.nn as nn
from models.s4.s4d import S4D  # 跟你训练一致


class S4InitialColorModel(nn.Module):
    def __init__(self, d_in=8, d_model=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([S4D(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 3))

    def forward(self, x):
        # x: [B, K, 8]
        h = self.in_proj(x)        # [B,K,d_model]
        h = h.transpose(1, 2)      # [B,d_model,K]
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        h = h.transpose(1, 2)      # [B,K,d_model]
        h = self.norm(h)
        return self.head(h[:, -1, :])  # [B,3]


def parse_csv_floats(s, n):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != n:
        raise ValueError(f"Expect {n} values, got {len(parts)}: {s}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="s4_initial_color.pt")
    ap.add_argument("--K", type=int, default=None, help="默认用 ckpt 里的 K；不填就自动读取")
    ap.add_argument("--rgb", required=True, help="例如: 120,80,60")
    ap.add_argument("--env", default=None,
                    help="可选: temp,hum,lux1,lux2,lux3 例如: 25.0,60.0,100,100,100 ; 不填则用训练均值")
    args = ap.parse_args()

    # ✅ PyTorch 2.6+ 需要 weights_only=False（因为 ckpt 里存了 numpy mean/std）
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean = ckpt["mean"]  # (8,)
    std = ckpt["std"]    # (8,)
    K = int(ckpt.get("K", 1)) if args.K is None else args.K

    rgb = parse_csv_floats(args.rgb, 3)  # (3,)

    if args.env is None:
        env = mean[3:].astype(np.float32)  # 用训练均值填 env (5,)
    else:
        env = parse_csv_floats(args.env, 5)

    x = np.concatenate([rgb, env], axis=0).astype(np.float32)  # (8,)

    # 组装成 [K,8]。你训练 K=1 的话，这里就是 1 行
    x_in = np.repeat(x[None, :], K, axis=0)  # [K,8]
    x_in = np.where(np.isfinite(x_in), x_in, mean[None, :]).astype(np.float32)

    # normalize
    x_n = (x_in - mean) / std
    x_n = np.nan_to_num(x_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = S4InitialColorModel(d_in=8).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_n[None, :, :]).to(device)).cpu().numpy()[0]  # (3,)

    pred_rgb = pred_n * std[:3] + mean[:3]
    print("Input RGB:", rgb)
    print("Env used :", env)
    print("Predicted initial RGB:", pred_rgb)


if __name__ == "__main__":
    main()
