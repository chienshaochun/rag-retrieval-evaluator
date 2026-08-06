"""Pure functions used to evaluate document-level retrieval rankings."""

from math import log2
from statistics import fmean


def _validate_k(k: int) -> None:
    """Require a positive cutoff for metrics calculated at K."""
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def _validate_k_values(k_values: list[int]) -> None:
    """Require at least one unique, positive cutoff."""
    if not k_values:
        raise ValueError("k_values must not be empty")
    if len(k_values) != len(set(k_values)):
        raise ValueError("k_values must not contain duplicates")

    for k in k_values:
        _validate_k(k)


def build_ranked_document_ids(
    ranked_chunk_ids: list[str],
    chunk_to_document: dict[str, str],
) -> list[str]:
    """Convert a chunk ranking into a first-occurrence document ranking.

    Each document is included only once, at the position where one of its
    chunks first appears. Every ranked chunk must exist in
    ``chunk_to_document``.
    """
    ranked_document_ids: list[str] = []
    seen_document_ids: set[str] = set()

    for chunk_id in ranked_chunk_ids:
        if chunk_id not in chunk_to_document:
            raise ValueError(f"Unknown chunk_id in ranking: {chunk_id!r}")

        document_id = chunk_to_document[chunk_id]
        if document_id in seen_document_ids:
            continue

        ranked_document_ids.append(document_id)
        seen_document_ids.add(document_id)

    return ranked_document_ids


def hit_at_k(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int,
) -> int:
    """Return 1 when at least one relevant document appears in the top K."""
    _validate_k(k)

    top_k_document_ids = ranked_document_ids[:k]
    return int(any(document_id in relevant_document_ids for document_id in top_k_document_ids))


def precision_at_k(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of the K ranking positions that are relevant."""
    _validate_k(k)

    top_k_document_ids = ranked_document_ids[:k]
    relevant_count = sum(
        document_id in relevant_document_ids for document_id in top_k_document_ids
    )
    return relevant_count / k


def recall_at_k(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of all relevant documents found in the top K."""
    _validate_k(k)
    if not relevant_document_ids:
        raise ValueError("relevant_document_ids must not be empty")

    top_k_document_ids = ranked_document_ids[:k]
    relevant_count = sum(
        document_id in relevant_document_ids for document_id in top_k_document_ids
    )
    return relevant_count / len(relevant_document_ids)


def reciprocal_rank(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant document."""
    if not relevant_document_ids:
        raise ValueError("relevant_document_ids must not be empty")

    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_document_ids:
            return 1 / rank

    return 0.0


def _discounted_gain(rank: int) -> float:
    """Return the binary-relevance discount applied at a one-based rank."""
    return 1 / log2(rank + 1)


def ndcg_at_k(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return binary normalized discounted cumulative gain at K."""
    _validate_k(k)
    if not relevant_document_ids:
        raise ValueError("relevant_document_ids must not be empty")

    top_k_document_ids = ranked_document_ids[:k]
    dcg = sum(
        _discounted_gain(rank)
        for rank, document_id in enumerate(top_k_document_ids, start=1)
        if document_id in relevant_document_ids
    )

    ideal_relevant_count = min(k, len(relevant_document_ids))
    idcg = sum(
        _discounted_gain(rank) for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg


def depth_coverage_at_k(
    ranked_document_ids: list[str],
    k: int,
) -> float:
    """Return how much of K can be filled by the retrieved unique documents."""
    _validate_k(k)

    unique_document_count = len(set(ranked_document_ids))
    return min(unique_document_count, k) / k


def evaluate_ranking(
    ranked_document_ids: list[str],
    relevant_document_ids: set[str],
    k_values: list[int],
) -> dict[str, dict[str, int | float] | float]:
    """Calculate every single-query metric for the requested cutoffs."""
    _validate_k_values(k_values)
    if not relevant_document_ids:
        raise ValueError("relevant_document_ids must not be empty")

    return {
        "hit_at_k": {
            str(k): hit_at_k(ranked_document_ids, relevant_document_ids, k)
            for k in k_values
        },
        "precision_at_k": {
            str(k): precision_at_k(ranked_document_ids, relevant_document_ids, k)
            for k in k_values
        },
        "recall_at_k": {
            str(k): recall_at_k(ranked_document_ids, relevant_document_ids, k)
            for k in k_values
        },
        "reciprocal_rank": reciprocal_rank(
            ranked_document_ids,
            relevant_document_ids,
        ),
        "ndcg_at_k": {
            str(k): ndcg_at_k(ranked_document_ids, relevant_document_ids, k)
            for k in k_values
        },
        "depth_coverage_at_k": {
            str(k): depth_coverage_at_k(ranked_document_ids, k) for k in k_values
        },
    }


def _mean_metric_at_k(
    query_metrics: list[dict[str, dict[str, int | float] | float]],
    metric_name: str,
    k: int,
) -> float:
    """Average one at-K metric across all query results."""
    k_label = str(k)
    values: list[float] = []

    for query_result in query_metrics:
        metrics_at_k = query_result.get(metric_name)
        if not isinstance(metrics_at_k, dict) or k_label not in metrics_at_k:
            raise ValueError(f"{metric_name} is missing k={k}")
        values.append(float(metrics_at_k[k_label]))

    return fmean(values)


def aggregate_metrics(
    query_metrics: list[dict[str, dict[str, int | float] | float]],
    k_values: list[int],
) -> dict[str, dict[str, float] | float]:
    """Macro-average single-query metrics across an evaluable dataset."""
    _validate_k_values(k_values)
    if not query_metrics:
        raise ValueError("query_metrics must not be empty")

    reciprocal_ranks: list[float] = []
    for query_result in query_metrics:
        value = query_result.get("reciprocal_rank")
        if not isinstance(value, (int, float)):
            raise ValueError("reciprocal_rank is missing or invalid")
        reciprocal_ranks.append(float(value))

    return {
        "hit_rate_at_k": {
            str(k): _mean_metric_at_k(query_metrics, "hit_at_k", k)
            for k in k_values
        },
        "macro_precision_at_k": {
            str(k): _mean_metric_at_k(query_metrics, "precision_at_k", k)
            for k in k_values
        },
        "macro_recall_at_k": {
            str(k): _mean_metric_at_k(query_metrics, "recall_at_k", k)
            for k in k_values
        },
        "mrr": fmean(reciprocal_ranks),
        "macro_ndcg_at_k": {
            str(k): _mean_metric_at_k(query_metrics, "ndcg_at_k", k)
            for k in k_values
        },
        "macro_depth_coverage_at_k": {
            str(k): _mean_metric_at_k(query_metrics, "depth_coverage_at_k", k)
            for k in k_values
        },
    }
