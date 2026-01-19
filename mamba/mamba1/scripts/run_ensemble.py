# scripts/run_ensemble.py
import argparse
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out_root", default="runs/ens")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for s in seeds:
        run_dir = out_root / f"seed{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["python", "scripts/train_mvp.py", "--config", args.config, "--run_dir", str(run_dir), "--seed", str(s)]
        print("\n>>>", " ".join(cmd))
        subprocess.check_call(cmd)

    print(f"\n[OK] ensemble trained under: {out_root}")

if __name__ == "__main__":
    main()
