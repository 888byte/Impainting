#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pigment_task.infer_pigment

Inference / evaluation for pigment fading -> original color (t0).

更新点（配合 方案A sequence 数据）：
- 观测颜色不再固定取 x0[:,1]，而是取 **mask 最后一个观测点** 作为 x_curr，用于 pred/retrieval 条件生成。
- 可选 Kalman RTS 平滑：把扩散输出的 t0 均值/方差当作观测，与 t1..tL-1 的观测序列一起做后向平滑，
  让预测 t0 更符合时间序列的平滑先验（创新点之一，可开关）。

Modes:
1) Evaluate on a test NPZ:
   python pigment_task/infer_pigment.py --ckpt ckpt/.../best_model.pt --test_npz data/.../test.npz --cond_method true
   python pigment_task/infer_pigment.py --ckpt ckpt/.../best_model.pt --test_npz data/.../test.npz --cond_method pred --num_samples 20
   python pigment_task/infer_pigment.py --ckpt ckpt/.../best_model.pt --test_npz data/.../test.npz --cond_method retrieval --library_npz data/standard_alignment/library_embeddings.npz

2) Predict one sample from RGB only (fallback L=2):
   python pigment_task/infer_pigment.py --ckpt ... --rgb "120,80,60" --cond_method pred --num_samples 30
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pigment_task.color_utils import LabNorm, rgb_to_lab, lab_to_rgb, delta_e2000
from pigment_task.dataset_pigment import PigmentNPZDataset
from pigment_task.diffusion import DiffusionConfig, DiffusionSchedule, p_sample_loop
from pigment_task.models.color_encoder import ColorEncoder, ColorEncoderConfig
from pigment_task.models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from pigment_task.models.pigment_denoiser import DenoiserConfig, MambaDenoiser
from pigment_task.models.spectral_encoder import ConditionerConfig, MultimodalConditioner


# =========================
# Small Kalman smoother (RTS) for random-walk model
# =========================

def _rts_smoother_random_walk(
    y: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """RTS smoother for x_t = x_{t-1} + w, y_t = x_t + v.

    y: (L,3) observed (Lab)
    R: (L,3) measurement variance per dim
    Q: (3,) process variance per dim

    Returns x_smooth: (L,3)
    """
    y = np.asarray(y, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64).reshape(3,)

    L = int(y.shape[0])
    x_f = np.zeros((L, 3), dtype=np.float64)
    P_f = np.zeros((L, 3), dtype=np.float64)

    # init with y0
    x = y[0].copy()
    P = np.maximum(R[0].copy(), 1e-6)

    for t in range(L):
        if t > 0:
            # predict
            P = P + Q
        # update
        S = P + np.maximum(R[t], 1e-6)
        K = P / S
        x = x + K * (y[t] - x)
        P = (1.0 - K) * P
        x_f[t] = x
        P_f[t] = P

    x_s = x_f.copy()
    P_s = P_f.copy()

    for t in range(L - 2, -1, -1):
        # random walk: F=I -> C = P_f[t] / (P_f[t] + Q)
        denom = P_f[t] + Q
        C = np.where(denom > 1e-12, P_f[t] / denom, 0.0)
        x_s[t] = x_f[t] + C * (x_s[t + 1] - x_f[t])
        P_s[t] = P_f[t] + C * (P_s[t + 1] - (P_f[t] + Q))

    return x_s


# =========================
# Checkpoint loading
# =========================

def _load_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    mod_cfg = cfg.get("modality", {}) if isinstance(cfg, dict) else {}
    cond_cfg = ConditionerConfig(
        use_raman=bool(mod_cfg.get("use_raman", False)),
        use_xrd=bool(mod_cfg.get("use_xrd", False)),
        raman_len=int(mod_cfg.get("raman_len", 1024)),
        xrd_len=int(mod_cfg.get("xrd_len", 2048)),
        d_model=int(mod_cfg.get("spec_d_model", 128)),
        n_layers=int(mod_cfg.get("spec_n_layers", 4)),
        dropout=float(mod_cfg.get("spec_dropout", 0.0)),
        raman_peak_dim=int(mod_cfg.get("raman_peak_dim", 0)),
        xrd_peak_dim=int(mod_cfg.get("xrd_peak_dim", 0)),
        use_fuse=bool(mod_cfg.get("use_fuse", True)),
    )
    conditioner = MultimodalConditioner(cond_cfg).to(device)
    if "conditioner" in ckpt:
        conditioner.load_state_dict(ckpt["conditioner"], strict=False)
    conditioner.eval()

    cond_dim = int(conditioner.cond_dim)

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    den_cfg = DenoiserConfig(
        in_channels=int(model_cfg.get("in_channels", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        cond_dim=cond_dim,
    )
    denoiser = MambaDenoiser(den_cfg).to(device)
    if "denoiser" in ckpt:
        denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    denoiser.eval()

    color_encoder: Optional[ColorEncoder] = None
    cond_predictor: Optional[ColorToSpecPredictor] = None
    if "color_encoder" in ckpt and "cond_predictor" in ckpt:
        mm_cfg = cfg.get("missing_modality", {}) if isinstance(cfg, dict) else {}
        ce_cfg = ColorEncoderConfig(
            in_dim=3,
            d_model=int(mm_cfg.get("color_d_model", cond_cfg.d_model)),
            hidden_dim=int(mm_cfg.get("color_hidden_dim", 256)),
            n_layers=int(mm_cfg.get("color_n_layers", 2)),
            dropout=float(mm_cfg.get("color_dropout", 0.0)),
        )
        color_encoder = ColorEncoder(ce_cfg).to(device)
        color_encoder.load_state_dict(ckpt["color_encoder"], strict=False)
        color_encoder.eval()

        cp_cfg = CondPredictorConfig(
            in_dim=ce_cfg.d_model,
            d_model=cond_cfg.d_model,
            use_raman=bool(cond_cfg.use_raman),
            use_xrd=bool(cond_cfg.use_xrd),
            hidden_dim=int(mm_cfg.get("pred_hidden_dim", 256)),
            n_layers=int(mm_cfg.get("pred_n_layers", 2)),
            dropout=float(mm_cfg.get("pred_dropout", 0.0)),
        )
        cond_predictor = ColorToSpecPredictor(cp_cfg).to(device)
        cond_predictor.load_state_dict(ckpt["cond_predictor"], strict=False)
        cond_predictor.eval()

    diff_cfg = DiffusionConfig(
        T=int(cfg.get("diffusion", {}).get("T", 200)),
        beta_0=float(cfg.get("diffusion", {}).get("beta_0", 1e-4)),
        beta_T=float(cfg.get("diffusion", {}).get("beta_T", 0.02)),
    )
    schedule = DiffusionSchedule(diff_cfg, device=device)

    return cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor


# =========================
# Condition building
# =========================

def _build_cond_from_pred_embeds(conditioner: MultimodalConditioner, embeds_pred: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if conditioner.cond_dim == 0:
        return None
    feats = []
    if conditioner.raman_enc is not None:
        feats.append(embeds_pred["raman"])
    if conditioner.xrd_enc is not None:
        feats.append(embeds_pred["xrd"])
    cond_cat = torch.cat(feats, dim=-1)
    return conditioner.fuse(cond_cat)


def _predict_embeds_from_rgb(
    x_curr_norm_lab: torch.Tensor,
    conditioner: MultimodalConditioner,
    color_encoder: ColorEncoder,
    cond_predictor: ColorToSpecPredictor,
) -> Dict[str, torch.Tensor]:
    zc = color_encoder(x_curr_norm_lab)
    pred = cond_predictor(zc)
    out: Dict[str, torch.Tensor] = {}
    if conditioner.raman_enc is not None:
        out["raman"] = pred["raman_emb"]
    if conditioner.xrd_enc is not None:
        out["xrd"] = pred["xrd_emb"]
    return out


def _load_library_npz(path: str) -> Dict[str, np.ndarray]:
    lib = np.load(path, allow_pickle=True)
    return {k: lib[k] for k in lib.files}


def _retrieval_raman_embed(
    z_color: torch.Tensor,
    lib_raman_emb: torch.Tensor,
    top_k: int = 5,
    temp: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-modal retrieval: query=z_color, keys=lib_raman_emb."""
    zq = F.normalize(z_color, dim=-1)
    zk = F.normalize(lib_raman_emb, dim=-1)
    sim = zq @ zk.T  # (B,M)

    k = min(int(top_k), sim.size(1))
    topv, topi = torch.topk(sim, k=k, dim=-1)
    w = F.softmax(topv / float(temp), dim=-1)  # (B,k)

    gathered = lib_raman_emb[topi]  # (B,k,d)
    r = torch.sum(gathered * w.unsqueeze(-1), dim=1)  # (B,d)
    return r, w, topi


def _retrieval_confidence(weights: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    w = torch.clamp(weights, eps, 1.0)
    ent = -torch.sum(w * torch.log(w), dim=-1)
    ent_norm = ent / np.log(weights.size(-1) + eps)
    return (1.0 - ent_norm).clamp(0.0, 1.0)


# =========================
# Sampling + uncertainty
# =========================

@torch.no_grad()
def _sample_with_confidence(
    denoiser: MambaDenoiser,
    schedule: DiffusionSchedule,
    x0_norm: torch.Tensor,
    mask: torch.Tensor,
    cond: Optional[torch.Tensor],
    num_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return (pred_mean_lab0, pred_std_lab0, info).

    pred_mean_lab0: (B,3) denormalized Lab
    pred_std_lab0:  (B,3) denormalized Lab std (per channel)
    """
    device = x0_norm.device
    lab_norm = LabNorm()

    x_obs = x0_norm * mask

    samples = []
    for _ in range(int(num_samples)):
        x_s = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())  # missing t0
    arr = np.stack(samples, axis=0)  # (S,B,3) normalized

    mean_norm = np.mean(arr, axis=0)
    std_norm = np.std(arr, axis=0)

    mean_lab = lab_norm.denormalize(mean_norm)
    # denormalize std (linear scaling)
    std_lab = std_norm * np.asarray([lab_norm.L_scale, lab_norm.ab_scale, lab_norm.ab_scale], dtype=np.float32)

    std_scalar = float(np.mean(np.linalg.norm(std_norm, axis=-1)))
    conf_diff = float(np.exp(-std_scalar))

    info = {
        "diffusion_std_norm_meanL2": std_scalar,
        "conf_diffusion": conf_diff,
        "num_samples": int(num_samples),
    }
    return mean_lab.astype(np.float32), std_lab.astype(np.float32), info


def _last_observed_index(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B,L,3) -> (B,) last index where observed==1"""
    B, L, _ = mask.shape
    obs = (mask.mean(dim=-1) > 0.5).to(torch.long)  # (B,L)
    idx_range = torch.arange(L, device=mask.device).view(1, L).expand(B, L)
    idx = torch.max(idx_range * obs, dim=1).values
    return idx


def _gather_last_observed(x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = _last_observed_index(mask)
    B = x0.shape[0]
    return x0[torch.arange(B, device=x0.device), idx]


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate_test(
    denoiser: MambaDenoiser,
    conditioner: MultimodalConditioner,
    schedule: DiffusionSchedule,
    test_npz: str,
    device: torch.device,
    cond_method: str,
    color_encoder: Optional[ColorEncoder] = None,
    cond_predictor: Optional[ColorToSpecPredictor] = None,
    library_npz: Optional[str] = None,
    retrieval_k: int = 5,
    retrieval_temp: float = 0.07,
    num_samples: int = 1,
    max_batches: int = 50,
    kalman_refine: bool = False,
    kalman_meas_std_lab: float = 1.0,
    kalman_process_std_lab: float = 2.0,
) -> Dict[str, float]:
    ds = PigmentNPZDataset(test_npz)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    lib_raman = None
    if cond_method == "retrieval":
        if library_npz is None:
            raise ValueError("cond_method=retrieval requires --library_npz")
        lib = _load_library_npz(library_npz)
        if "raman_emb" not in lib:
            raise ValueError("library_npz must contain 'raman_emb'")
        lib_raman = torch.from_numpy(lib["raman_emb"].astype(np.float32)).to(device)

    lab_norm = LabNorm()

    de_list = []
    conf_list = []

    for bi, batch in enumerate(dl):
        if bi >= int(max_batches):
            break
        x0 = batch["x0"].to(device)
        mask = batch["mask"].to(device)
        x_curr = _gather_last_observed(x0, mask)  # normalized Lab

        # build condition
        conf_ret = None
        if conditioner.cond_dim == 0:
            cond = None
        elif cond_method == "true":
            cond = conditioner(
                batch.get("raman", None).to(device) if "raman" in batch else None,
                batch.get("xrd", None).to(device) if "xrd" in batch else None,
                raman_peaks=batch.get("raman_peaks", None).to(device) if "raman_peaks" in batch else None,
                xrd_peaks=batch.get("xrd_peaks", None).to(device) if "xrd_peaks" in batch else None,
                return_embeds=False,
            )
        elif cond_method == "pred":
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_method=pred requires ckpt with color_encoder+cond_predictor")
            embeds_pred = _predict_embeds_from_rgb(x_curr, conditioner, color_encoder, cond_predictor)
            cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
        elif cond_method == "retrieval":
            if color_encoder is None or lib_raman is None:
                raise ValueError("cond_method=retrieval requires ckpt color_encoder and library_npz")
            zc = color_encoder(x_curr)
            raman_emb, w, _ = _retrieval_raman_embed(zc, lib_raman, top_k=int(retrieval_k), temp=float(retrieval_temp))
            embeds_pred: Dict[str, torch.Tensor] = {}
            if conditioner.raman_enc is not None:
                embeds_pred["raman"] = raman_emb
            if conditioner.xrd_enc is not None:
                if cond_predictor is None:
                    embeds_pred["xrd"] = torch.zeros_like(raman_emb)
                else:
                    tmp = cond_predictor(zc)
                    embeds_pred["xrd"] = tmp["xrd_emb"]
            cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
            conf_ret = _retrieval_confidence(w).detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown cond_method: {cond_method}")

        if int(num_samples) <= 1:
            x_obs = x0 * mask
            x_sample = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask, cond=cond)
            pred0 = lab_norm.denormalize(x_sample[:, 0, :].detach().cpu().numpy())
            gt0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
        else:
            pred0, std0, info = _sample_with_confidence(denoiser, schedule, x0, mask, cond, num_samples=int(num_samples))
            gt0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())

            if kalman_refine:
                # Build y sequence: [pred0] + observed labs from input (denormalized)
                x0_den = lab_norm.denormalize(x0.detach().cpu().numpy())  # (B,L,3)
                mask_np = mask.detach().cpu().numpy()
                B, L, _ = x0_den.shape
                meas_std = float(kalman_meas_std_lab)
                proc_std = float(kalman_process_std_lab)
                Q = np.asarray([proc_std**2, proc_std**2, proc_std**2], dtype=np.float64)

                refined = []
                for i in range(B):
                    y = x0_den[i].copy()
                    # replace t0 with predicted
                    y[0] = pred0[i]

                    # R: t0 use model variance, observed steps use meas_std^2
                    R = np.ones((L, 3), dtype=np.float64) * (meas_std**2)
                    R[0] = np.maximum(std0[i] ** 2, 1e-6)
                    # if some steps are unobserved (rare), increase R
                    unobs = (mask_np[i].mean(axis=-1) < 0.5)
                    R[unobs] = 1e6

                    xs = _rts_smoother_random_walk(y, R, Q)
                    refined.append(xs[0])
                pred0 = np.stack(refined, axis=0).astype(np.float32)

            conf_list.extend([info["conf_diffusion"]] * pred0.shape[0])

        # deltaE
        for p, g in zip(pred0, gt0):
            de_list.append(float(delta_e2000(p, g)))

        if conf_ret is not None:
            conf_list.extend(list(conf_ret))

    out = {
        "deltaE2000_mean": float(np.mean(de_list)) if de_list else float("nan"),
        "deltaE2000_std": float(np.std(de_list)) if de_list else float("nan"),
    }
    if conf_list:
        out["confidence_mean"] = float(np.mean(conf_list))
        out["confidence_std"] = float(np.std(conf_list))
    return out


# =========================
# CLI
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")

    # evaluation
    ap.add_argument("--test_npz", type=str, default="")
    ap.add_argument("--max_batches", type=int, default=50)

    # single query
    ap.add_argument("--rgb", type=str, default="", help='Single RGB like "120,80,60"')

    # missing-modality inference
    ap.add_argument("--cond_method", type=str, default="true", choices=["true", "pred", "retrieval"])
    ap.add_argument("--library_npz", type=str, default="", help="NPZ with standard Raman library embeddings")
    ap.add_argument("--retrieval_k", type=int, default=5)
    ap.add_argument("--retrieval_temp", type=float, default=0.07)

    # uncertainty
    ap.add_argument("--num_samples", type=int, default=1, help="Diffusion samples per query (>=2 enables diffusion uncertainty proxy)")

    # optional Kalman RTS refinement
    ap.add_argument("--kalman_refine", action="store_true", help="Refine predicted t0 using RTS smoother over full sequence")
    ap.add_argument("--kalman_rts", action="store_true", help="Alias of --kalman_refine (backward compatible)")
    ap.add_argument("--kalman_meas_std_lab", type=float, default=1.0, help="Measurement std (Lab units) for observed steps when using kalman_refine")
    ap.add_argument("--kalman_process_std_lab", type=float, default=2.0, help="Process std (Lab units) for random-walk dynamics when using kalman_refine")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = _load_ckpt(args.ckpt, device)

    if args.test_npz:
        stats = evaluate_test(
            denoiser,
            conditioner,
            schedule,
            test_npz=args.test_npz,
            device=device,
            cond_method=args.cond_method,
            color_encoder=color_encoder,
            cond_predictor=cond_predictor,
            library_npz=args.library_npz if args.library_npz else None,
            retrieval_k=int(args.retrieval_k),
            retrieval_temp=float(args.retrieval_temp),
            num_samples=int(args.num_samples),
            max_batches=int(args.max_batches),
            kalman_refine=bool(args.kalman_refine or args.kalman_rts),
            kalman_meas_std_lab=float(args.kalman_meas_std_lab),
            kalman_process_std_lab=float(args.kalman_process_std_lab),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.rgb:
        rgb = np.array([float(x) for x in args.rgb.split(",")], dtype=np.float32)
        if rgb.size != 3:
            raise ValueError("--rgb must be 'R,G,B'")

        lab = rgb_to_lab(rgb[None, :])[0]
        lab_norm = LabNorm()
        x_curr = lab_norm.normalize(lab[None, :]).astype(np.float32)  # (1,3)

        # Dummy L=2 sample
        x0 = np.stack([lab, lab], axis=0)[None, :, :]
        x0n = lab_norm.normalize(x0).astype(np.float32)
        mask = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)

        x0_t = torch.from_numpy(x0n).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        x_curr_t = torch.from_numpy(x_curr).to(device)

        # cond
        conf_ret = None
        if conditioner.cond_dim == 0:
            cond = None
        elif args.cond_method == "true":
            raise ValueError("Single RGB mode does not have spectra; use --cond_method pred or retrieval")
        elif args.cond_method == "pred":
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_method=pred requires ckpt with color_encoder+cond_predictor")
            embeds_pred = _predict_embeds_from_rgb(x_curr_t, conditioner, color_encoder, cond_predictor)
            cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
        else:  # retrieval
            if not args.library_npz:
                raise ValueError("cond_method=retrieval requires --library_npz")
            if color_encoder is None:
                raise ValueError("cond_method=retrieval requires ckpt with color_encoder")
            lib = _load_library_npz(args.library_npz)
            lib_raman = torch.from_numpy(lib["raman_emb"].astype(np.float32)).to(device)
            zc = color_encoder(x_curr_t)
            raman_emb, w, _ = _retrieval_raman_embed(zc, lib_raman, top_k=int(args.retrieval_k), temp=float(args.retrieval_temp))
            embeds_pred: Dict[str, torch.Tensor] = {}
            if conditioner.raman_enc is not None:
                embeds_pred["raman"] = raman_emb
            if conditioner.xrd_enc is not None:
                if cond_predictor is None:
                    embeds_pred["xrd"] = torch.zeros_like(raman_emb)
                else:
                    tmp = cond_predictor(zc)
                    embeds_pred["xrd"] = tmp["xrd_emb"]
            cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
            conf_ret = float(_retrieval_confidence(w)[0].item())

        # sampling
        if int(args.num_samples) <= 1:
            x_obs = x0_t * mask_t
            x_s = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask_t, cond=cond)
            pred_lab0 = lab_norm.denormalize(x_s[:, 0, :].detach().cpu().numpy())[0]
            std_lab0 = None
            info = {"num_samples": 1, "conf_diffusion": None, "diffusion_std_norm_meanL2": None}
        else:
            pred_lab0_b, std_lab0_b, info = _sample_with_confidence(denoiser, schedule, x0_t, mask_t, cond, num_samples=int(args.num_samples))
            pred_lab0 = pred_lab0_b[0]
            std_lab0 = std_lab0_b[0]

        pred_rgb0 = lab_to_rgb(pred_lab0[None, :])[0]

        out = {
            "input_rgb_current": rgb.tolist(),
            "pred_lab_original": pred_lab0.tolist(),
            "pred_rgb_original": pred_rgb0.tolist(),
            "pred_lab_std": std_lab0.tolist() if std_lab0 is not None else None,
            "confidence_diffusion": info.get("conf_diffusion", None),
            "diffusion_std_norm_meanL2": info.get("diffusion_std_norm_meanL2", None),
            "confidence_retrieval": conf_ret,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise ValueError("Provide either --test_npz or --rgb")


if __name__ == "__main__":
    main()
