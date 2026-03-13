"""Bridge modules for condition construction and posterior conditioning."""

from .condition_builder import (
    build_cond_from_pred_embeds,
    build_posterior_condition,
    build_pred_condition,
    build_retrieval_condition,
    build_true_condition,
    gather_last_observed,
    last_observed_index,
)
from .posterior_head import PosteriorHead, PosteriorHeadConfig
from .prototype_bank import PrototypeBank, build_prototype_bank
