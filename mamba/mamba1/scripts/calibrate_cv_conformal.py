# scripts/calibrate_cv_conformal.py
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config, lab_unnorm
from train_mvp import SliceDataset, collate_fn, MambaRegressor

def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    Din = int(ckpt["Din"])
    cfg = ckpt.get("cfg", {})
    model = MambaRegressor(
        Din=Din,
        d_model=int(cfg.get("d_model", 128)),
        n_layers=int(cfg.get("n_layers", 4)),
        d_state=int(cfg.get("d_state", 16)),
        d_conv=int(cfg.get("d_conv", 4)),
        expand=int(cfg.get("expand", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--loso_root", default="runs/loso")
    ap.add_argument("--out_dir", default="runs/cv_calib")
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--slices_per_seq", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    loso_root = Path(args.loso_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(cfg.get("batch_size", 32))

    systems = sorted([p.name for p in loso_root.iterdir() if p.is_dir()])
    if not systems:
        raise FileNotFoundError(f"No LOSO folders under {loso_root}")

    per_system = {}
    for sys_name in systems:
        ckpts = sorted(list((loso_root / sys_name).glob("seed*/best.pt")))
        if not ckpts:
            print(f"[WARN] skip {sys_name}: no ckpts")
            continue

        # dataset only of this heldout system
        ds = SliceDataset(str(seq_path), [sys_name], cfg, slices_per_seq=args.slices_per_seq)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

        abs_err_all = []
        with torch.no_grad():
            for ckpt_path in ckpts:
                model = load_model(ckpt_path, device)
                for X, lengths, y, _ in loader:
                    X, lengths, y = X.to(device), lengths.to(device), y.to(device)
                    mu = model(X, lengths)
                    mu_raw = lab_unnorm(mu.cpu().numpy())
                    y_raw = lab_unnorm(y.cpu().numpy())
                    abs_err_all.append(np.abs(y_raw - mu_raw))

        abs_err_all = np.concatenate(abs_err_all, axis=0)  # (num_samples,3)
        q_marg = np.quantile(abs_err_all, args.q, axis=0)
        q_joint = float(np.quantile(np.max(abs_err_all, axis=1), args.q))
        per_system[sys_name] = {
            "q_marginal_rawLab": q_marg.tolist(),
            "q_joint_rawLab": q_joint,
            "num_err_samples": int(abs_err_all.shape[0]),
            "num_models": int(len(ckpts)),
        }
        print(f"[OK] {sys_name}: q_marg={q_marg}  q_joint={q_joint:.3f}  samples={abs_err_all.shape[0]}")

    # conservative max across systems
    q_m_max = np.max(np.stack([np.array(v["q_marginal_rawLab"], dtype=np.float32) for v in per_system.values()], axis=0), axis=0)
    q_j_max = float(np.max([v["q_joint_rawLab"] for v in per_system.values()]))

    calib = {
        "method": "LOSO-CV-Conformal (max over systems)",
        "q": args.q,
        "q_marginal_rawLab": q_m_max.tolist(),
        "q_joint_rawLab": q_j_max,
        "per_system": per_system,
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote conservative calibration to: {out_dir / 'calibration.json'}")
    print(json.dumps({k: calib[k] for k in ["q","q_marginal_rawLab","q_joint_rawLab","method"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
