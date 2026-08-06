"""Calculate numeric differences between reranker metrics."""

from typing import cast


def _metric_deltas_at_k(
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    metric_name: str,
    k_values: list[int],
) -> dict[str, float]:
    before_at_k = cast(dict[str, int | float], before_metrics[metric_name])
    after_at_k = cast(dict[str, int | float], after_metrics[metric_name])
    return {
        str(k): float(after_at_k[str(k)]) - float(before_at_k[str(k)])
        for k in k_values
    }


def calculate_query_metric_deltas(
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    k_values: list[int],
) -> dict[str, dict[str, float] | float]:
    """Subtract before from after for every single-query quality metric."""
    return {
        "hit_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "hit_at_k", k_values
        ),
        "precision_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "precision_at_k", k_values
        ),
        "recall_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "recall_at_k", k_values
        ),
        "reciprocal_rank": float(after_metrics["reciprocal_rank"])
        - float(before_metrics["reciprocal_rank"]),
        "ndcg_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "ndcg_at_k", k_values
        ),
    }


def calculate_aggregate_metric_deltas(
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    k_values: list[int],
) -> dict[str, dict[str, float] | float]:
    """Subtract before from after for every dataset-level quality metric."""
    return {
        "hit_rate_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "hit_rate_at_k", k_values
        ),
        "macro_precision_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "macro_precision_at_k", k_values
        ),
        "macro_recall_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "macro_recall_at_k", k_values
        ),
        "mrr": float(after_metrics["mrr"]) - float(before_metrics["mrr"]),
        "macro_ndcg_at_k": _metric_deltas_at_k(
            before_metrics, after_metrics, "macro_ndcg_at_k", k_values
        ),
    }
