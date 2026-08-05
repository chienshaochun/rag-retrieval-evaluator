"""Core evaluation utilities for RAG retrieval results."""

from .evaluation import evaluate_matched_query
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
from .pairing import pair_query_records
from .validation import (
    GroundTruthValidationError,
    RetrievalResultsValidationError,
    validate_ground_truth,
    validate_retrieval_results,
)

__all__ = [
    "evaluate_matched_query",
    "aggregate_metrics",
    "build_ranked_document_ids",
    "depth_coverage_at_k",
    "evaluate_ranking",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "pair_query_records",
    "GroundTruthValidationError",
    "RetrievalResultsValidationError",
    "validate_ground_truth",
    "validate_retrieval_results",
]
