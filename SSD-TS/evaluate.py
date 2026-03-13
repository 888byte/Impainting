import argparse
import json
import torch

from evaluation.ablations import main as ablation_main
from inference.pipeline import evaluate_test, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='test', choices=['test', 'palette', 'mine'])
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--test_npz', type=str, default='')
    ap.add_argument('--max_batches', type=int, default=50)
    ap.add_argument('--cond_method', type=str, default='auto')
    ap.add_argument('--library_npz', type=str, default='')
    ap.add_argument('--prototype_bank', type=str, default='')
    ap.add_argument('--retrieval_k', type=int, default=5)
    ap.add_argument('--retrieval_temp', type=float, default=0.07)
    ap.add_argument('--num_samples', type=int, default=1)
    ap.add_argument('--kalman_refine', action='store_true')
    ap.add_argument('--kalman_rts', action='store_true')
    ap.add_argument('--kalman_meas_std_lab', type=float, default=1.0)
    ap.add_argument('--kalman_process_std_lab', type=float, default=2.0)
    ap.add_argument('--n_colors', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_rows', type=int, default=20000)
    ap.add_argument('--out_csv', type=str, default='')
    args, unknown = ap.parse_known_args()
    if args.mode == 'test':
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)
        stats = evaluate_test(
            bundle=bundle,
            test_npz=args.test_npz,
            device=device,
            cond_method=args.cond_method,
            library_npz=args.library_npz if args.library_npz else None,
            retrieval_k=args.retrieval_k,
            retrieval_temp=args.retrieval_temp,
            num_samples=args.num_samples,
            max_batches=args.max_batches,
            kalman_refine=bool(args.kalman_refine or args.kalman_rts),
            kalman_meas_std_lab=args.kalman_meas_std_lab,
            kalman_process_std_lab=args.kalman_process_std_lab,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    import sys
    sys.argv = ['evaluate.py', '--ckpt', args.ckpt, '--device', args.device, '--mode', args.mode, '--cond_method', args.cond_method, '--library_npz', args.library_npz, '--prototype_bank', args.prototype_bank, '--retrieval_k', str(args.retrieval_k), '--retrieval_temp', str(args.retrieval_temp), '--num_samples', str(args.num_samples), '--n_colors', str(args.n_colors), '--seed', str(args.seed), '--test_npz', args.test_npz, '--max_rows', str(args.max_rows), '--out_csv', args.out_csv] + unknown
    ablation_main()


if __name__ == '__main__':
    main()
