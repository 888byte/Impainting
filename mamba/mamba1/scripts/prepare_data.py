\
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import (
    load_config,
    open_workbook,
    read_two_columns_numeric,
    interp_to_grid,
    normalize_spec,
    fista_nonneg_lasso,
    parse_color_log_records,
    make_monotonic_time,
    rgb_series_to_lab,
)

def parse_raman_sheet_name(name: str):
    # example: "66 铅丹+密陀僧_532"
    import re
    m = re.match(r"(\d+)\s+(.+?)_(532|785)$", name.strip())
    if not m:
        return None
    env = int(m.group(1))
    system = m.group(2).strip()
    laser = int(m.group(3))
    return env, system, laser

def build_raman_dictionary(cfg):
    raw_dir = Path(cfg["raw_dir"])
    lib_path = raw_dir / cfg["raman_library"]
    exp_path = raw_dir / cfg["raman_exp"]

    wb_lib = open_workbook(str(lib_path))
    wb_exp = open_workbook(str(exp_path))

    # Get target grids from your experimental Raman files
    # 532 grid: use any 532 sheet (here: "66 密陀僧_532" exists in your file)
    # 785 grid: use the only 785 sheet (here: "76 铅丹_785")
    grid_532, _ = read_two_columns_numeric(wb_exp, "66 密陀僧_532")
    grid_785, _ = read_two_columns_numeric(wb_exp, "76 铅丹_785")

    # Select library sheets by naming convention
    sheets_532 = [s for s in wb_lib.sheetnames if ("532" in s and ("785" not in s and "780" not in s))]
    sheets_785 = [s for s in wb_lib.sheetnames if ("785" in s or "780" in s)]

    # Build D matrices: (M, K)
    D532 = []
    for s in tqdm(sheets_532, desc="Building D532"):
        x, y = read_two_columns_numeric(wb_lib, s)
        yi = normalize_spec(interp_to_grid(x, y, grid_532))
        D532.append(yi)
    D532 = np.stack(D532, axis=1).astype(np.float32)

    D785 = []
    for s in tqdm(sheets_785, desc="Building D785"):
        x, y = read_two_columns_numeric(wb_lib, s)
        yi = normalize_spec(interp_to_grid(x, y, grid_785))
        D785.append(yi)
    D785 = np.stack(D785, axis=1).astype(np.float32)

    out = Path(cfg["processed_dir"]) / "raman_dict.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        grid_532=grid_532, grid_785=grid_785,
        D532=D532, D785=D785,
        names_532=np.array(sheets_532, dtype=object),
        names_785=np.array(sheets_785, dtype=object),
    )
    print(f"[OK] saved Raman dictionary to: {out}")
    return out


def compute_raman_coeffs(cfg, dict_npz_path: Path):
    raw_dir = Path(cfg["raw_dir"])
    exp_path = raw_dir / cfg["raman_exp"]
    wb_exp = open_workbook(str(exp_path))

    d = np.load(dict_npz_path, allow_pickle=True)
    grid_532 = d["grid_532"].astype(np.float32)
    grid_785 = d["grid_785"].astype(np.float32)
    D532 = d["D532"].astype(np.float32)
    D785 = d["D785"].astype(np.float32)
    K532 = D532.shape[1]
    K785 = D785.shape[1]

    sparse_lam = float(cfg["sparse_lam"])
    fista_iters = int(cfg["fista_iters"])

    coeffs = {}  # (env, system) -> vec(Ktotal)

    for sheet in tqdm(wb_exp.sheetnames, desc="Sparse-encoding Raman (exp)"):
        parsed = parse_raman_sheet_name(sheet)
        if not parsed:
            continue
        env, system, laser = parsed
        x, y = read_two_columns_numeric(wb_exp, sheet)
        if laser == 532:
            r = normalize_spec(interp_to_grid(x, y, grid_532))
            a = fista_nonneg_lasso(D532, r, lam=sparse_lam, n_iter=fista_iters)
            vec = np.concatenate([a, np.zeros(K785, dtype=np.float32)], axis=0)
        else:
            r = normalize_spec(interp_to_grid(x, y, grid_785))
            a = fista_nonneg_lasso(D785, r, lam=sparse_lam, n_iter=fista_iters)
            vec = np.concatenate([np.zeros(K532, dtype=np.float32), a], axis=0)
        # normalize coefficients to sum=1 (mixture-like feature)
        s = float(vec.sum())
        if s > 1e-8:
            vec = (vec / s).astype(np.float32)
        coeffs[(env, system)] = vec

    out = Path(cfg["processed_dir"]) / "raman_coeffs.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # store as parallel arrays for easy loading
    keys = np.array([[k[0], k[1]] for k in coeffs.keys()], dtype=object)
    vals = np.stack(list(coeffs.values()), axis=0).astype(np.float32)
    np.savez_compressed(out, keys=keys, vals=vals)
    print(f"[OK] saved Raman coefficients to: {out}")
    return out


def build_sequences(cfg, coeffs_npz_path: Path):
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    coeffs_data = np.load(coeffs_npz_path, allow_pickle=True)
    keys = coeffs_data["keys"]
    vals = coeffs_data["vals"]
    coeffs = {(int(k[0]), str(k[1])): vals[i] for i, k in enumerate(keys)}

    sequences = []  # list of dicts

    for env_str, log_name in cfg["color_logs"].items():
        env = int(env_str)
        log_path = raw_dir / log_name
        recs = parse_color_log_records(str(log_path))
        if cfg.get("drop_first_color_record", True) and len(recs) > 1:
            recs = recs[1:]  # drop calibration-like first record

        hours = [r["hour"] for r in recs]
        t = make_monotonic_time(hours)

        temp = np.array([r["temp"] for r in recs], dtype=np.float32)
        hum = np.array([r["hum"] for r in recs], dtype=np.float32)

        mapping = cfg["no_to_system"][env]
        for no_str, system in mapping.items():
            no = int(no_str)
            rgb = np.array([r[f"rgb{no}"] for r in recs], dtype=np.uint8)  # (T,3)
            lab = rgb_series_to_lab(rgb)  # (T,3) raw scale

            # get raman coef (static per sequence in this MVP)
            key = (env, system)
            if key not in coeffs:
                raise KeyError(f"Missing Raman coefficient for (env={env}, system='{system}'). "
                               f"Check config.no_to_system or Raman sheet naming.")
            a = coeffs[key].astype(np.float32)

            sequences.append({
                "env": env,
                "no": no,
                "system": system,
                "t": t,
                "lab": lab,
                "temp": temp,
                "hum": hum,
                "a": a,
            })

    # Save
    out = processed_dir / "dataset_sequences.npz"
    N = len(sequences)
    T = sequences[0]["lab"].shape[0]
    K = sequences[0]["a"].shape[0]

    env = np.array([s["env"] for s in sequences], dtype=np.int32)
    no = np.array([s["no"] for s in sequences], dtype=np.int32)
    system = np.array([s["system"] for s in sequences], dtype=object)
    t = np.stack([s["t"] for s in sequences], axis=0).astype(np.float32)
    lab = np.stack([s["lab"] for s in sequences], axis=0).astype(np.float32)
    temp = np.stack([s["temp"] for s in sequences], axis=0).astype(np.float32)
    hum = np.stack([s["hum"] for s in sequences], axis=0).astype(np.float32)
    a = np.stack([s["a"] for s in sequences], axis=0).astype(np.float32)  # (N,K), static

    np.savez_compressed(out, env=env, no=no, system=system, t=t, lab=lab, temp=temp, hum=hum, a=a)
    print(f"[OK] saved sequences to: {out}  (N={N}, T={T}, K={K})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dict_npz = build_raman_dictionary(cfg)
    coeffs_npz = compute_raman_coeffs(cfg, dict_npz)
    build_sequences(cfg, coeffs_npz)


if __name__ == "__main__":
    main()
