"""Core evaluation utilities for RAG retrieval results."""

from .comparison import (
    calculate_aggregate_metric_deltas,
    calculate_query_metric_deltas,
)
from .evaluation import evaluate_matched_queries, evaluate_matched_query
from .failure_analysis import generate_failure_tags, summarize_failure_tags
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
from .pipeline import run_evaluation_pipeline
from .validation import (
    GroundTruthValidationError,
    RetrievalResultsValidationError,
    validate_ground_truth,
    validate_retrieval_results,
)

__all__ = [
    "calculate_aggregate_metric_deltas",
    "calculate_query_metric_deltas",
    "evaluate_matched_query",
    "evaluate_matched_queries",
    "generate_failure_tags",
    "summarize_failure_tags",
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
    "run_evaluation_pipeline",
    "GroundTruthValidationError",
    "RetrievalResultsValidationError",
    "validate_ground_truth",
    "validate_retrieval_results",
]
