# scripts/train_mvp.py
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import load_config, lab_norm, lab_unnorm
from mamba_ssm import Mamba


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_sys(s) -> str:
    if s is None:
        return ""
    x = str(s)
    x = x.replace("＋", "+").replace("\u3000", "")
    x = x.replace("\t", "").replace("\n", "").replace("\r", "")
    x = x.replace(" ", "")
    return x.strip()


def delta_e76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab1 - lab2, axis=-1)


class SliceDataset(Dataset):
    """
    Randomly slice a suffix [t_obs..T), reverse it (late->early) and predict Lab0.

    token feature:
      [lab_norm(3), a_or_a_seq(K), dt_log(1), lab_mask(1), raman_mask(1)]
    """

    def __init__(self, sequences_npz: str, systems_keep: list[str], cfg: dict, slices_per_seq: int):
        d = np.load(sequences_npz, allow_pickle=True)
        self.env = d["env"]
        self.no = d["no"]
        self.system = np.array([norm_sys(s) for s in d["system"].astype(object).tolist()], dtype=object)
        self.t = d["t"]
        self.lab = d["lab"]
        self.a = d["a"].astype(np.float32)  # (N,K)
        self.a_seq = d["a_seq"].astype(np.float32) if "a_seq" in d else None  # (N,T,K) optional
        self.cfg = cfg

        keep_set = set(norm_sys(s) for s in systems_keep)
        keep_mask = np.array([s in keep_set for s in self.system], dtype=bool)
        self.idx_map = np.where(keep_mask)[0].tolist()
        if len(self.idx_map) == 0:
            uniq_data = sorted(set(self.system.tolist()))
            uniq_keep = sorted(list(keep_set))
            raise ValueError(
                "No sequences left after split!\n"
                f"- keep_set({len(uniq_keep)}): {uniq_keep}\n"
                f"- data_systems({len(uniq_data)}): {uniq_data}\n"
            )

        self.slices_per_seq = int(slices_per_seq)
        self.T = self.lab.shape[1]
        self.K = self.a.shape[1]

        self.t_obs_min = int(cfg.get("t_obs_min", 1))
        self.use_last_n_at_train = int(cfg.get("use_last_n_at_train", 0))

    def __len__(self):
        return len(self.idx_map) * self.slices_per_seq

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx + 12345)
        seq_i = self.idx_map[idx % len(self.idx_map)]
        T = self.T

        # label: Lab at t0
        y = self.lab[seq_i, 0:1, :]  # (1,3)
        y_n = lab_norm(y).squeeze(0).astype(np.float32)  # (3,)

        # suffix start
        t_obs = rng.randint(self.t_obs_min, T)
        t_seq = self.t[seq_i]
        lab_seq = self.lab[seq_i]

        t_in = t_seq[t_obs:]
        lab_in = lab_seq[t_obs:]

        if self.use_last_n_at_train > 0 and len(t_in) > 2:
            n = rng.randint(2, min(self.use_last_n_at_train, len(t_in)) + 1)
            t_in = t_in[-n:]
            lab_in = lab_in[-n:]
            t_obs = T - n

        # reverse: late -> early
        t_rev = t_in[::-1].copy()
        lab_rev = lab_in[::-1].copy()

        dt = np.zeros((len(t_rev),), dtype=np.float32)
        if len(t_rev) >= 2:
            dt[1:] = (t_rev[:-1] - t_rev[1:]).astype(np.float32)
        dt_log = np.log1p(dt).reshape(-1, 1).astype(np.float32)

        lab_n = lab_norm(lab_rev).astype(np.float32)  # (L,3)

        if self.a_seq is not None:
            a_in = self.a_seq[seq_i, t_obs:, :]              # (L,K)
            if self.use_last_n_at_train > 0 and a_in.shape[0] != lab_n.shape[0]:
                a_in = a_in[-lab_n.shape[0]:, :]
            a_feat = a_in[::-1].copy().astype(np.float32)    # reverse align
        else:
            a_vec = self.a[seq_i]
            a_feat = np.repeat(a_vec.reshape(1, -1), repeats=len(t_rev), axis=0).astype(np.float32)

        lab_mask = np.ones((len(t_rev), 1), dtype=np.float32)
        raman_mask = np.ones((len(t_rev), 1), dtype=np.float32)

        x = np.concatenate([lab_n, a_feat, dt_log, lab_mask, raman_mask], axis=1).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "len": torch.tensor([x.shape[0]], dtype=torch.long),
            "y": torch.from_numpy(y_n),
            "meta": (int(self.env[seq_i]), int(self.no[seq_i]), str(self.system[seq_i])),
        }


def collate_fn(batch):
    lengths = torch.cat([b["len"] for b in batch], dim=0)
    maxL = int(lengths.max().item())
    Din = batch[0]["x"].shape[1]
    B = len(batch)

    X = torch.zeros((B, maxL, Din), dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        X[i, :L] = b["x"]

    y = torch.stack([b["y"] for b in batch], dim=0).float()
    metas = [b["meta"] for b in batch]
    return X, lengths, y, metas


class MambaRegressor(nn.Module):
    def __init__(self, Din: int, d_model=128, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_proj = nn.Linear(Din, d_model)
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
            }))
        self.out_norm = nn.LayerNorm(d_model)
        self.head_mu = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, X, lengths):
        h = self.in_proj(X)
        for blk in self.blocks:
            h = h + blk["mamba"](blk["norm"](h))
        h = self.out_norm(h)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        h_last = h.gather(1, idx).squeeze(1)
        return self.head_mu(h_last)


def split_by_system(systems_all: list[str], seed: int = 42, n_val: int = 2, n_test: int = 1):
    uniq = sorted(list(set(norm_sys(s) for s in systems_all if norm_sys(s))))
    if len(uniq) < 3:
        return uniq, [uniq[0]], [uniq[0]]
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_test = min(n_test, max(1, len(uniq) - 2))
    n_val = min(n_val, max(1, len(uniq) - n_test - 1))
    test_sys = uniq[:n_test]
    val_sys = uniq[n_test:n_test + n_val]
    train_sys = uniq[n_test + n_val:]
    if len(train_sys) == 0:
        train_sys = uniq
    return train_sys, val_sys, test_sys


def parse_csv_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [norm_sys(x) for x in s.split(",") if norm_sys(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", default="runs/mvp")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--train_systems", default="")
    ap.add_argument("--val_systems", default="")
    ap.add_argument("--test_systems", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42)) if args.seed < 0 else int(args.seed)
    set_seed(seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    d = np.load(seq_path, allow_pickle=True)
    systems_all = [norm_sys(s) for s in d["system"].astype(object).tolist()]

    # allow explicit split (for LOSO / ensemble)
    train_sys_arg = parse_csv_list(args.train_systems)
    val_sys_arg   = parse_csv_list(args.val_systems)
    test_sys_arg  = parse_csv_list(args.test_systems)

    if train_sys_arg or val_sys_arg or test_sys_arg:
        uniq = sorted(list(set(norm_sys(s) for s in systems_all if norm_sys(s))))
        def ensure(lst, name):
            for x in lst:
                if x not in uniq:
                    raise ValueError(f"{name} contains unknown system: {x}. known={uniq}")
        ensure(train_sys_arg, "train_systems")
        ensure(val_sys_arg, "val_systems")
        ensure(test_sys_arg, "test_systems")

        train_sys = train_sys_arg if train_sys_arg else uniq
        val_sys   = val_sys_arg   if val_sys_arg   else train_sys
        test_sys  = test_sys_arg  if test_sys_arg  else []
    else:
        train_sys, val_sys, test_sys = split_by_system(
            systems_all,
            seed=seed,
            n_val=int(cfg.get("n_val_systems", 2)),
            n_test=int(cfg.get("n_test_systems", 1)),
        )

    slices = int(cfg.get("slices_per_seq_per_epoch", 64))
    train_ds = SliceDataset(str(seq_path), train_sys, cfg, slices_per_seq=slices)
    val_ds = SliceDataset(str(seq_path), val_sys, cfg, slices_per_seq=max(64, slices))

    bs = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    Din = next(iter(train_loader))[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MambaRegressor(
        Din=Din,
        d_model=int(cfg.get("d_model", 128)),
        n_layers=int(cfg.get("n_layers", 4)),
        d_state=int(cfg.get("d_state", 16)),
        d_conv=int(cfg.get("d_conv", 4)),
        expand=int(cfg.get("expand", 2)),
    ).to(device)

    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-2))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps = int(cfg.get("steps", 2500))
    eval_every = int(cfg.get("eval_every", 100))
    patience = int(cfg.get("early_stop_patience", 8))

    huber_beta = float(cfg.get("huber_beta", 0.05))
    loss_fn = nn.SmoothL1Loss(beta=huber_beta)

    best_de = float("inf")
    best_path = run_dir / "best.pt"
    bad_rounds = 0

    it = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            X, lengths, y, _ = next(it)
        except StopIteration:
            it = iter(train_loader)
            X, lengths, y, _ = next(it)

        X, lengths, y = X.to(device), lengths.to(device), y.to(device)
        model.train()
        mu = model(X, lengths)
        loss = loss_fn(mu, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                de_list = []
                val_loss_list = []
                for Xv, Lv, yv, _ in val_loader:
                    Xv, Lv, yv = Xv.to(device), Lv.to(device), yv.to(device)
                    muv = model(Xv, Lv)
                    val_loss_list.append(float(loss_fn(muv, yv).item()))
                    y_raw = lab_unnorm(yv.cpu().numpy())
                    mu_raw = lab_unnorm(muv.cpu().numpy())
                    de_list.extend(delta_e76(y_raw, mu_raw).tolist())
                val_de = float(np.mean(de_list)) if len(de_list) else float("nan")
                val_loss = float(np.mean(val_loss_list))

            print(f"step {step:4d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | val_ΔE76={val_de:.3f}")

            if (val_de + 1e-9) < best_de:
                best_de = val_de
                bad_rounds = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "Din": Din,
                        "seed": seed,
                        "split": {"train": train_sys, "val": val_sys, "test": test_sys},
                        "best": {"step": step, "val_de": best_de, "val_loss": val_loss},
                    },
                    best_path
                )
            else:
                bad_rounds += 1

            if bad_rounds >= patience:
                print(f"[EARLY STOP] no val_ΔE improvement for {patience} evals. stopping at step={step}.")
                break

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_ckpt": str(best_path),
                "best_de": best_de,
                "seed": seed,
                "split": {"train": train_sys, "val": val_sys, "test": test_sys},
            },
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    print(f"[OK] saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
