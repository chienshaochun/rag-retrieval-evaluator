"""Generate deterministic diagnostic tags for reranker query results."""

from math import isclose
from typing import cast

DELTA_ABS_TOLERANCE = 1e-12

DIAGNOSTIC_TAGS = (
    "no_relevant_candidate",
    "partial_relevant_candidate_coverage",
    "top_k_total_miss",
    "top_k_incomplete_coverage",
    "persistent_top_k_miss",
    "reranker_regression",
    "reranker_rescue",
    "reranker_improvement",
    "reranker_unchanged",
)

FAILURE_CASE_TAGS = frozenset(
    {
        "no_relevant_candidate",
        "partial_relevant_candidate_coverage",
        "top_k_total_miss",
        "top_k_incomplete_coverage",
        "persistent_top_k_miss",
        "reranker_regression",
    }
)


def generate_failure_tags(
    query_result: dict[str, object],
    primary_k: int,
) -> list[str]:
    """Classify candidate coverage, top-K outcome, and reranker direction."""
    if primary_k <= 0:
        raise ValueError("primary_k must be a positive integer")

    relevant_document_ids = set(
        cast(list[str], query_result["relevant_document_ids"])
    )
    before_result = cast(dict[str, object], query_result["before_reranking"])
    after_result = cast(dict[str, object], query_result["after_reranking"])
    before_ranked_document_ids = cast(
        list[str], before_result["ranked_document_ids"]
    )
    after_ranked_document_ids = cast(list[str], after_result["ranked_document_ids"])

    candidate_relevant_ids = set(before_ranked_document_ids) & relevant_document_ids
    before_top_k_relevant_ids = (
        set(before_ranked_document_ids[:primary_k]) & relevant_document_ids
    )
    after_top_k_relevant_ids = (
        set(after_ranked_document_ids[:primary_k]) & relevant_document_ids
    )

    tags: list[str] = []

    if not candidate_relevant_ids:
        tags.append("no_relevant_candidate")
    elif len(candidate_relevant_ids) < len(relevant_document_ids):
        tags.append("partial_relevant_candidate_coverage")

    if not after_top_k_relevant_ids:
        tags.append("top_k_total_miss")
    elif len(after_top_k_relevant_ids) < len(relevant_document_ids):
        tags.append("top_k_incomplete_coverage")

    if not before_top_k_relevant_ids and not after_top_k_relevant_ids:
        tags.append("persistent_top_k_miss")

    metric_deltas = cast(dict[str, object], query_result["metric_deltas"])
    ndcg_deltas = cast(dict[str, int | float], metric_deltas["ndcg_at_k"])
    k_label = str(primary_k)
    if k_label not in ndcg_deltas:
        raise ValueError("primary_k must be included in k_values")
    ndcg_delta = float(ndcg_deltas[k_label])

    if ndcg_delta < -DELTA_ABS_TOLERANCE:
        tags.append("reranker_regression")

    if not before_top_k_relevant_ids and after_top_k_relevant_ids:
        tags.append("reranker_rescue")

    if ndcg_delta > DELTA_ABS_TOLERANCE:
        tags.append("reranker_improvement")
    elif isclose(ndcg_delta, 0.0, rel_tol=0.0, abs_tol=DELTA_ABS_TOLERANCE):
        tags.append("reranker_unchanged")

    return tags


def summarize_failure_tags(
    query_results: list[dict[str, object]],
    primary_k: int,
) -> dict[str, object]:
    """Summarize diagnostic tags and build an inspectable failure-case list."""
    if not query_results:
        raise ValueError("query_results must not be empty")
    if primary_k <= 0:
        raise ValueError("primary_k must be a positive integer")

    tag_counts = {tag: 0 for tag in DIAGNOSTIC_TAGS}
    failure_cases: list[dict[str, object]] = []
    k_label = str(primary_k)

    for query_result in query_results:
        failure_tags = cast(list[str], query_result["failure_tags"])
        for tag in failure_tags:
            if tag not in tag_counts:
                raise ValueError(f"unknown failure tag: {tag!r}")
            tag_counts[tag] += 1

        if not FAILURE_CASE_TAGS.intersection(failure_tags):
            continue

        before_result = cast(dict[str, object], query_result["before_reranking"])
        after_result = cast(dict[str, object], query_result["after_reranking"])
        before_metrics = cast(dict[str, object], before_result["metrics"])
        after_metrics = cast(dict[str, object], after_result["metrics"])
        before_ndcg = cast(dict[str, int | float], before_metrics["ndcg_at_k"])
        after_ndcg = cast(dict[str, int | float], after_metrics["ndcg_at_k"])
        metric_deltas = cast(dict[str, object], query_result["metric_deltas"])
        ndcg_deltas = cast(dict[str, int | float], metric_deltas["ndcg_at_k"])

        if (
            k_label not in before_ndcg
            or k_label not in after_ndcg
            or k_label not in ndcg_deltas
        ):
            raise ValueError("primary_k must be included in k_values")

        before_ranked_document_ids = cast(
            list[str], before_result["ranked_document_ids"]
        )
        after_ranked_document_ids = cast(
            list[str], after_result["ranked_document_ids"]
        )
        failure_cases.append(
            {
                "query_id": query_result["query_id"],
                "query_text": query_result["query_text"],
                "relevant_document_ids": query_result["relevant_document_ids"],
                "failure_tags": list(failure_tags),
                "primary_k": primary_k,
                "before_top_k_document_ids": before_ranked_document_ids[:primary_k],
                "after_top_k_document_ids": after_ranked_document_ids[:primary_k],
                "before_ndcg_at_primary_k": before_ndcg[k_label],
                "after_ndcg_at_primary_k": after_ndcg[k_label],
                "ndcg_delta_at_primary_k": ndcg_deltas[k_label],
            }
        )

    evaluated_query_count = len(query_results)
    failure_case_count = len(failure_cases)
    tag_rates = {
        tag: count / evaluated_query_count for tag, count in tag_counts.items()
    }

    return {
        "evaluated_query_count": evaluated_query_count,
        "failure_case_count": failure_case_count,
        "failure_case_rate": failure_case_count / evaluated_query_count,
        "tag_counts": tag_counts,
        "tag_rates": tag_rates,
        "failure_cases": failure_cases,
    }
