import argparse
import os
import torch

from bridge.prototype_bank import build_prototype_bank
from inference.pipeline import load_checkpoint
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from utils.config_utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--ckpt', type=str, default='')
    ap.add_argument('--output', type=str, default='')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--index_csv', type=str, default='')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.ckpt:
        bundle = load_checkpoint(args.ckpt, device)
        conditioner = bundle['conditioner']
    else:
        mod_cfg = cfg.get('modality', {})
        cond_cfg = ConditionerConfig(
            use_raman=bool(mod_cfg.get('use_raman', False)),
            use_xrd=bool(mod_cfg.get('use_xrd', False)),
            raman_len=int(mod_cfg.get('raman_len', 1024)),
            xrd_len=int(mod_cfg.get('xrd_len', 2048)),
            d_model=int(mod_cfg.get('spec_d_model', 128)),
            n_layers=int(mod_cfg.get('spec_n_layers', 4)),
            dropout=float(mod_cfg.get('spec_dropout', 0.0)),
            raman_peak_dim=int(mod_cfg.get('raman_peak_dim', 0)),
            xrd_peak_dim=int(mod_cfg.get('xrd_peak_dim', 0)),
            use_fuse=bool(mod_cfg.get('use_fuse', True)),
        )
        conditioner = MultimodalConditioner(cond_cfg).to(device)
        pretrained = cfg.get('pretrained', {})
        if pretrained.get('raman_encoder_ckpt'):
            sd = torch.load(pretrained['raman_encoder_ckpt'], map_location='cpu')
            conditioner.load_state_dict(sd.get('conditioner', sd), strict=False)
    bank = build_prototype_bank(
        npz_path=cfg['data']['train_npz'],
        conditioner=conditioner,
        device=device,
        index_csv=args.index_csv or cfg.get('data', {}).get('train_index', ''),
    )
    out_path = args.output or cfg.get('bridge', {}).get('prototype_bank', {}).get('path') or os.path.join(os.path.dirname(args.config), 'prototype_bank.npz')
    bank.save(out_path)
    print(out_path)


if __name__ == '__main__':
    main()
