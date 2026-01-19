# scripts/run_loso_ensemble.py
import argparse
import subprocess
from pathlib import Path
import numpy as np

from utils import load_config

def norm_sys(s) -> str:
    if s is None:
        return ""
    x = str(s)
    x = x.replace("＋", "+").replace("\u3000", "")
    x = x.replace("\t", "").replace("\n", "").replace("\r", "")
    x = x.replace(" ", "")
    return x.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out_root", default="runs/loso")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_path = Path(cfg["processed_dir"]) / "dataset_sequences.npz"
    d = np.load(seq_path, allow_pickle=True)
    systems_all = [norm_sys(s) for s in d["system"].astype(object).tolist()]
    uniq = sorted(list(set([s for s in systems_all if s])))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for hold in uniq:
        train_sys = [s for s in uniq if s != hold]
        # val 就用 train_sys（2 个系统时也不会为空）
        val_sys = train_sys[:]

        hold_dir = out_root / hold
        hold_dir.mkdir(parents=True, exist_ok=True)

        for sd in seeds:
            run_dir = hold_dir / f"seed{sd}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python", "scripts/train_mvp.py",
                "--config", args.config,
                "--run_dir", str(run_dir),
                "--seed", str(sd),
                "--train_systems", ",".join(train_sys),
                "--val_systems", ",".join(val_sys),
                "--test_systems", hold,   # 仅记录在 ckpt 里
            ]
            print("\n>>>", " ".join(cmd))
            subprocess.check_call(cmd)

    print(f"\n[OK] LOSO models saved under: {out_root}")

if __name__ == "__main__":
    main()
