"""Legacy wrapper for `python -m pigment_task.infer_pigment`."""
from bridge.condition_builder import (
    build_cond_from_pred_embeds as _build_cond_from_pred_embeds,
    load_library_npz as _load_library_npz,
    predict_embeds_from_rgb as _predict_embeds_from_rgb,
    retrieval_confidence as _retrieval_confidence,
    retrieval_raman_embed as _retrieval_raman_embed,
)
from inference.pipeline import load_checkpoint as _load_ckpt, main
from inference.uncertainty import sample_with_confidence as _sample_with_confidence

if __name__ == '__main__':
    main()
