import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_config, lab_unnorm
from train_mvp import SliceDataset, collate_fn, MambaRegressor, delta_e76


def coverage_joint(mu_raw, rad_raw, y_raw):
    lo = mu_raw - rad_raw
    hi = mu_raw + rad_raw
    covered = np.all((y_raw >= lo) & (y_raw <= hi), axis=-1)
    return float(np.mean(covered))


def coverage_marginal(mu_raw, rad_raw, y_raw):
    lo = mu_raw - rad_raw
    hi = mu_raw + rad_raw
    covered_dim = (y_raw >= lo) & (y_raw <= hi)
    return [float(np.mean(covered_dim[:, i])) for i in range(3)]


def get_preds(model, loader, device):
    mus_n, ys_n = [], []
    de_list = []
    model.eval()
    with torch.no_grad():
        for X, lengths, y, _ in loader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            mu = model(X, lengths)
            mu_n = mu.detach().cpu().numpy()
            y_n = y.detach().cpu().numpy()

            mu_raw = lab_unnorm(mu_n)
            y_raw = lab_unnorm(y_n)
            de_list.extend(delta_e76(y_raw, mu_raw).tolist())

            mus_n.append(mu_n)
            ys_n.append(y_n)

    mus_n = np.concatenate(mus_n, axis=0)
    ys_n = np.concatenate(ys_n, axis=0)
    return mus_n, ys_n, float(np.mean(de_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run_dir", default="runs/mvp_eval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    Din = int(ckpt["Din"])
    split = ckpt["split"]

    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    bs = int(cfg.get("batch_size", 32))

    val_ds = SliceDataset(str(seq_path), split["val"], cfg, slices_per_seq=512)
    test_ds = SliceDataset(str(seq_path), split["test"], cfg, slices_per_seq=512)

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    q = float(cfg.get("conformal_q", 0.9))

    # val residuals
    mu_v_n, y_v_n, val_de = get_preds(model, val_loader, device)
    mu_v_raw = lab_unnorm(mu_v_n)
    y_v_raw = lab_unnorm(y_v_n)
    abs_err_v = np.abs(y_v_raw - mu_v_raw)

    q_marg = np.quantile(abs_err_v, q, axis=0)
    q_joint = float(np.quantile(np.max(abs_err_v, axis=1), q))

    # test
    mu_t_n, y_t_n, test_de = get_preds(model, test_loader, device)
    mu_t_raw = lab_unnorm(mu_t_n)
    y_t_raw = lab_unnorm(y_t_n)

    rad_conf_marg = np.broadcast_to(q_marg.reshape(1, 3), mu_t_raw.shape)
    rad_conf_joint = np.full_like(mu_t_raw, q_joint)

    out = {
        "test_systems": split["test"],
        "val_systems": split["val"],
        "val_mean_ΔE76": val_de,
        "test_mean_ΔE76": test_de,
        "conformal_q": q,
        "conformal_q_marginal_rawLab": q_marg.tolist(),
        "conformal_q_joint_rawLab": q_joint,
        "coverage_conformal_joint": coverage_joint(mu_t_raw, rad_conf_joint, y_t_raw),
        "coverage_conformal_marginal": coverage_marginal(mu_t_raw, rad_conf_marg, y_t_raw),
    }

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "calibration.json").write_text(
        json.dumps({"q": q, "q_marginal_rawLab": q_marg.tolist(), "q_joint_rawLab": q_joint}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
