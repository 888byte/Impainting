# scripts/predict_ensemble.py
import argparse
from pathlib import Path
import json
import numpy as np
import torch

from utils import load_config, lab_norm, lab_unnorm
from train_mvp import MambaRegressor

def build_single_input(seq_npz_path: str, env: int, no: int, use_last_n: int = 0):
    d = np.load(seq_npz_path, allow_pickle=True)
    env_all = d["env"]; no_all = d["no"]; t_all = d["t"]; lab_all = d["lab"]
    system_all = d["system"].astype(object)
    a_seq_all = d["a_seq"] if "a_seq" in d else None
    a_all = d["a"]

    idx = None
    for i in range(len(env_all)):
        if int(env_all[i]) == env and int(no_all[i]) == no:
            idx = i
            break
    if idx is None:
        raise ValueError(f"Sequence not found for env={env}, no={no}")

    t = t_all[idx].astype(np.float32)
    lab = lab_all[idx].astype(np.float32)
    system = str(system_all[idx])

    if use_last_n and use_last_n > 1:
        t = t[-use_last_n:]
        lab = lab[-use_last_n:]
        if a_seq_all is not None:
            a_seq = a_seq_all[idx].astype(np.float32)[-use_last_n:]
        else:
            a_seq = None
    else:
        a_seq = a_seq_all[idx].astype(np.float32) if a_seq_all is not None else None

    t_rev = t[::-1].copy()
    lab_rev = lab[::-1].copy()

    dt = np.zeros((len(t_rev),), dtype=np.float32)
    if len(t_rev) >= 2:
        dt[1:] = (t_rev[:-1] - t_rev[1:]).astype(np.float32)
    dt_log = np.log1p(dt).reshape(-1, 1).astype(np.float32)

    lab_n = lab_norm(lab_rev).astype(np.float32)

    if a_seq is not None:
        a_feat = a_seq[::-1].copy().astype(np.float32)
    else:
        a_vec = a_all[idx].astype(np.float32)
        a_feat = np.repeat(a_vec.reshape(1, -1), repeats=len(t_rev), axis=0).astype(np.float32)

    lab_mask = np.ones((len(t_rev), 1), dtype=np.float32)
    raman_mask = np.ones((len(t_rev), 1), dtype=np.float32)

    x = np.concatenate([lab_n, a_feat, dt_log, lab_mask, raman_mask], axis=1).astype(np.float32)
    return x, system, lab[0:1]

def load_calib(calib_json: str):
    obj = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    q_m = np.array(obj["q_marginal_rawLab"], dtype=np.float32).reshape(3,)
    q_j = float(obj["q_joint_rawLab"])
    return q_m, q_j

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
    ap.add_argument("--ens_root", default="runs/ens")
    ap.add_argument("--calib_json", default="runs/cv_calib/calibration.json")
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--no", type=int, required=True)
    ap.add_argument("--use_last_n", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"

    x, system, lab0_true = build_single_input(str(seq_path), args.env, args.no, args.use_last_n)

    ens_root = Path(args.ens_root)
    ckpts = sorted(list(ens_root.glob("seed*/best.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found: {ens_root}/seed*/best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(p, device) for p in ckpts]

    X = torch.from_numpy(x).unsqueeze(0).to(device)
    lengths = torch.tensor([x.shape[0]], dtype=torch.long).to(device)

    mus = []
    with torch.no_grad():
        for m in models:
            mu = m(X, lengths)
            mu_raw = lab_unnorm(mu.cpu().numpy()).reshape(3)
            mus.append(mu_raw)

    mu_ens = np.mean(np.stack(mus, axis=0), axis=0)
    mu_std = np.std(np.stack(mus, axis=0), axis=0)  # epistemic proxy

    q_m, q_j = load_calib(args.calib_json)

    print("=== Ensemble Predict initial Lab0 ===")
    print(f"env={args.env} no={args.no} system={system}  used_last_n={args.use_last_n if args.use_last_n else 'ALL'}")
    print(f"mu_ens(Lab) = {mu_ens}")
    print(f"mu_std(Lab) = {mu_std}  (model spread; not calibrated)")
    print(f"true Lab0   = {lab0_true.reshape(3)}  (proxy label: earliest in log after cleanup)")
    print("\n--- CV-Conformal intervals (recommended for unseen system) ---")
    print(f"marginal radius = {q_m}")
    print(f"marginal interval = [{mu_ens - q_m}, {mu_ens + q_m}]")
    print(f"joint radius = {q_j}")
    print(f"joint interval = [{mu_ens - q_j}, {mu_ens + q_j}]")

if __name__ == "__main__":
    main()
