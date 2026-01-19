# scripts/eval_ensemble.py
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config, lab_unnorm
from train_mvp import SliceDataset, collate_fn, MambaRegressor, delta_e76

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
    return model, ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ens_root", default="runs/ens")
    ap.add_argument("--out_json", default="runs/ens_eval.json")
    ap.add_argument("--slices_per_seq", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    ens_root = Path(args.ens_root)

    ckpts = sorted(list(ens_root.glob("seed*/best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ens_root}/seed*/best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use split from the first ckpt
    first = torch.load(ckpts[0], map_location="cpu")
    split = first["split"]
    test_systems = split.get("test", [])
    if not test_systems:
        raise ValueError("ckpt split has no test systems; run normal split training first.")

    test_ds = SliceDataset(str(seq_path), test_systems, cfg, slices_per_seq=args.slices_per_seq)
    bs = int(cfg.get("batch_size", 32))
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    models = []
    for p in ckpts:
        m, _ = load_model(p, device)
        models.append((p, m))

    # eval each + ensemble
    per_model = {}
    de_all_ens = []

    with torch.no_grad():
        for X, lengths, y, _ in loader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            y_raw = lab_unnorm(y.cpu().numpy())

            mus = []
            for p, m in models:
                mu = m(X, lengths)
                mu_raw = lab_unnorm(mu.cpu().numpy())
                mus.append(mu_raw)

                # per-model accumulate
                de = delta_e76(y_raw, mu_raw)
                per_model.setdefault(str(p), []).append(de)

            mu_ens = np.mean(np.stack(mus, axis=0), axis=0)
            de_ens = delta_e76(y_raw, mu_ens)
            de_all_ens.append(de_ens)

    # summarize
    for k in list(per_model.keys()):
        per_model[k] = float(np.mean(np.concatenate(per_model[k], axis=0)))
    mean_de_ens = float(np.mean(np.concatenate(de_all_ens, axis=0)))

    out = {
        "test_systems": test_systems,
        "num_models": len(models),
        "mean_ΔE76_per_model": per_model,
        "mean_ΔE76_ensemble": mean_de_ens,
    }
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
