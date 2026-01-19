import argparse
from pathlib import Path
import json
import numpy as np
import torch

from utils import load_config, lab_norm, lab_unnorm
from train_mvp import MambaRegressor


def build_single_input(seq_npz_path: str, env: int, no: int, use_last_n: int = 0):
    d = np.load(seq_npz_path, allow_pickle=True)

    env_all = d["env"]
    no_all = d["no"]
    t_all = d["t"]
    lab_all = d["lab"]
    a_all = d["a"]
    system_all = d["system"].astype(object)

    idx = None
    for i in range(len(env_all)):
        if int(env_all[i]) == env and int(no_all[i]) == no:
            idx = i
            break
    if idx is None:
        raise ValueError(f"Sequence not found for env={env}, no={no}")

    t = t_all[idx].astype(np.float32)
    lab = lab_all[idx].astype(np.float32)
    a = a_all[idx].astype(np.float32)
    system = str(system_all[idx])

    if use_last_n and use_last_n > 1:
        t = t[-use_last_n:]
        lab = lab[-use_last_n:]

    t_rev = t[::-1].copy()
    lab_rev = lab[::-1].copy()

    dt = np.zeros((len(t_rev),), dtype=np.float32)
    if len(t_rev) >= 2:
        dt[1:] = (t_rev[:-1] - t_rev[1:]).astype(np.float32)
    dt_log = np.log1p(dt).reshape(-1, 1).astype(np.float32)

    lab_n = lab_norm(lab_rev).astype(np.float32)
    a_rep = np.repeat(a.reshape(1, -1), repeats=len(t_rev), axis=0).astype(np.float32)

    lab_mask = np.ones((len(t_rev), 1), dtype=np.float32)
    raman_mask = np.ones((len(t_rev), 1), dtype=np.float32)

    x = np.concatenate([lab_n, a_rep, dt_log, lab_mask, raman_mask], axis=1).astype(np.float32)
    return x, system, lab[0:1]


def load_conformal(calib_dir: str):
    p = Path(calib_dir) / "calibration.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    q_m = np.array(obj["q_marginal_rawLab"], dtype=np.float32).reshape(3,)
    q_j = float(obj["q_joint_rawLab"])
    q = float(obj.get("q", 0.9))
    return {"q": q, "q_marg": q_m, "q_joint": q_j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", type=int, required=True)
    ap.add_argument("--no", type=int, required=True)
    ap.add_argument("--use_last_n", type=int, default=0)
    ap.add_argument("--calib_from", default="runs/mvp_eval", help="dir containing calibration.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    x, system, lab0_true = build_single_input(str(seq_path), args.env, args.no, args.use_last_n)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    Din = int(ckpt["Din"])

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

    X = torch.from_numpy(x).unsqueeze(0).to(device)
    lengths = torch.tensor([x.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        mu = model(X, lengths)

    mu_raw = lab_unnorm(mu.cpu().numpy()).reshape(3)
    conf = load_conformal(args.calib_from)

    print("=== Predict initial Lab0 ===")
    print(f"env={args.env}  no={args.no}  system={system}")
    print(f"used_last_n={args.use_last_n if args.use_last_n else 'ALL'}")
    print(f"mu(Lab) = {mu_raw}")
    print(f"true Lab0 = {lab0_true.reshape(3)}  (proxy label: earliest in log after cleanup)")

    if conf is None:
        print("\n(no conformal calibration found; run eval_mvp.py to generate calibration.json)")
        return

    q_m = conf["q_marg"]
    q_j = conf["q_joint"]
    print(f"\n--- Uncertainty (conformal, q={conf['q']}) ---")
    print(f"marginal radius (raw Lab) = {q_m}")
    print(f"marginal interval = [{mu_raw - q_m}, {mu_raw + q_m}]")
    print(f"joint radius (raw Lab, max-norm) = {q_j}")
    print(f"joint interval = [{mu_raw - q_j}, {mu_raw + q_j}]")

if __name__ == "__main__":
    main()
