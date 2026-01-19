import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ✅ 这一行请按你官方仓库 example.py 里的 import 写法一致
from models.s4.s4d import S4D


class InitialColorDataset(Dataset):
    """
    data: [9, T, 8] = RGB(3)+env(5)
    训练样本构造：
      - 从同一条序列里随机取一个“现在时刻”t
      - 输入：最近 K 个点 (t-K+1 .. t)
      - 输出：该序列最开始的 RGB (time=0)
    """
    def __init__(self, npy_path, K=1, n_samples_per_series=2000):
        x = np.load(npy_path).astype(np.float32)  # [9,T,8]
        self.K = K

        # ✅ mean/std 用 1D (8,)
        mean = np.nanmean(x, axis=(0, 1)).astype(np.float32)          # (8,)
        std  = (np.nanstd(x, axis=(0, 1)) + 1e-6).astype(np.float32)  # (8,)

        # ✅ 用 mean 填 NaN/Inf
        x = np.where(np.isfinite(x), x, mean[None, None, :]).astype(np.float32)

        self.x = x
        self.N, self.T, self.D = x.shape
        self.mean = mean
        self.std = std

        # ✅ 一定要有 indices
        self.indices = []
        for n in range(self.N):
            for _ in range(n_samples_per_series):
                t = np.random.randint(self.K - 1, self.T)  # t >= K-1
                self.indices.append((n, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        n, t = self.indices[idx]
        seq = self.x[n]  # [T,8]

        x_in = seq[t - self.K + 1 : t + 1]  # [K,8]
        y0 = seq[0, :3]                     # [3]

        # 归一化
        x_in = (x_in - self.mean) / self.std
        y0   = (y0   - self.mean[:3]) / self.std[:3]

        # 双保险
        x_in = np.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y0   = np.nan_to_num(y0,   nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return torch.from_numpy(x_in), torch.from_numpy(y0)

class S4InitialColorModel(nn.Module):
    def __init__(self, d_in=8, d_model=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([S4D(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),  # 输出 RGB
        )

    def forward(self, x):
        # x: [B, K, 8]
        h = self.in_proj(x)          # [B, K, d_model]
        h = h.transpose(1, 2)        # -> [B, d_model, K]  (S4D 常用格式：B,H,L)

        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out   # ✅ 兼容 (y, state)

        h = h.transpose(1, 2)        # -> [B, K, d_model]
        h = self.norm(h)
        h_last = h[:, -1, :]         # [B, d_model]
        return self.head(h_last)     # [B, 3]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data/color_series.npy")
    ap.add_argument("--K", type=int, default=1, help="用几个最近点当输入，K=1 就是只用当前颜色")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples_per_series", type=int, default=2000)
    ap.add_argument("--save", default="s4_initial_color.pt")
    args = ap.parse_args()

    ds = InitialColorDataset(args.data, K=args.K, n_samples_per_series=args.samples_per_series)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = S4InitialColorModel(d_in=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs + 1):
        loss_sum = 0.0
        for x_in, y0 in dl:
            x_in = x_in.to(device)
            y0 = y0.to(device)

            pred = model(x_in)
            loss = loss_fn(pred, y0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()

        if ep % 10 == 0:
            print(f"epoch {ep} loss {loss_sum/len(dl):.6f}")

    torch.save({"model": model.state_dict(), "mean": ds.mean, "std": ds.std, "K": args.K}, args.save)
    print("saved:", args.save)


if __name__ == "__main__":
    main()
