# scripts/upgrade_a_seq.py
import argparse
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

def topk_l1(v: np.ndarray, k: int, eps: float = 1e-8):
    if k <= 0 or k >= v.shape[-1]:
        s = np.sum(np.clip(v, 0, None))
        return np.clip(v, 0, None) / (s + eps)
    vv = np.clip(v, 0, None)
    idx = np.argpartition(vv, -k)[-k:]
    out = np.zeros_like(vv)
    out[idx] = vv[idx]
    s = out.sum()
    out = out / (s + eps)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_npz", default="")
    ap.add_argument("--out_npz", default="")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--pair_mode", default="minmax", choices=["minmax", "env_pair"])
    ap.add_argument("--init_env", type=int, default=66)
    ap.add_argument("--changed_env", type=int, default=76)
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["processed_dir"])
    in_npz = Path(args.in_npz) if args.in_npz else (processed_dir / "dataset_sequences.npz")
    out_npz = Path(args.out_npz) if args.out_npz else in_npz  # inplace overwrite

    d = np.load(in_npz, allow_pickle=True)
    keys = list(d.keys())
    data = {k: d[k] for k in keys}

    assert "a" in data, "dataset_sequences.npz must contain 'a' (N,K)"
    assert "t" in data and "lab" in data and "env" in data and "system" in data, "missing required fields"

    a = data["a"].astype(np.float32)          # (N,K)  (your existing sparse coeffs)
    t = data["t"].astype(np.float32)          # (N,T)
    env = data["env"].astype(np.int32)        # (N,)
    system = np.array([norm_sys(s) for s in data["system"].astype(object).tolist()], dtype=object)

    N, T = t.shape
    K = a.shape[1]
    a_seq = np.zeros((N, T, K), dtype=np.float32)

    # build per-system endpoints: a_init, a_change
    uniq_sys = sorted(list(set(system.tolist())))
    sys2end = {}

    for s in uniq_sys:
        idxs = np.where(system == s)[0]
        envs = env[idxs]

        if args.pair_mode == "env_pair":
            init_e, chg_e = args.init_env, args.changed_env
        else:
            init_e, chg_e = int(envs.min()), int(envs.max())

        init_idxs = idxs[envs == init_e]
        chg_idxs  = idxs[envs == chg_e]

        if len(init_idxs) == 0:
            init_idxs = idxs
        if len(chg_idxs) == 0:
            chg_idxs = idxs

        a_init = a[init_idxs].mean(axis=0)
        a_chg  = a[chg_idxs].mean(axis=0)

        if args.topk > 0:
            a_init = topk_l1(a_init, args.topk)
            a_chg  = topk_l1(a_chg, args.topk)

        sys2end[s] = (a_init.astype(np.float32), a_chg.astype(np.float32), init_e, chg_e)

    # interpolate over time for each sequence
    for i in range(N):
        s = system[i]
        a_init, a_chg, init_e, chg_e = sys2end[s]
        ti = t[i]  # (T,)
        t0, t1 = float(ti.min()), float(ti.max())
        denom = (t1 - t0) if (t1 - t0) > 1e-8 else 1.0
        p = (ti - t0) / denom  # 0..1

        # linear interpolation in coeff space
        for j in range(T):
            aj = (1.0 - p[j]) * a_init + p[j] * a_chg
            if args.topk > 0:
                aj = topk_l1(aj, args.topk)
            a_seq[i, j] = aj.astype(np.float32)

    data["a_seq"] = a_seq
    np.savez_compressed(out_npz, **data)
    print(f"[OK] wrote a_seq to: {out_npz}  shape={a_seq.shape}  (N={N},T={T},K={K})")

if __name__ == "__main__":
    main()
