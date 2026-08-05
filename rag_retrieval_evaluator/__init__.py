"""Core evaluation utilities for RAG retrieval results."""

from .metrics import (
    aggregate_metrics,
    build_ranked_document_ids,
    depth_coverage_at_k,
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .validation import GroundTruthValidationError, validate_ground_truth

__all__ = [
    "aggregate_metrics",
    "build_ranked_document_ids",
    "depth_coverage_at_k",
    "evaluate_ranking",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "GroundTruthValidationError",
    "validate_ground_truth",
]
