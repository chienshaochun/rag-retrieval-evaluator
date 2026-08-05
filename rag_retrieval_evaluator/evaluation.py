"""Evaluate validated and paired retrieval records."""

from typing import cast

from .comparison import (
    calculate_aggregate_metric_deltas,
    calculate_query_metric_deltas,
)
from .failure_analysis import generate_failure_tags, summarize_failure_tags
from .metrics import aggregate_metrics, build_ranked_document_ids, evaluate_ranking


def _ranking_chunk_ids(ranking: object) -> list[str]:
    ranking_items = cast(list[dict[str, object]], ranking)
    return [cast(str, item["chunk_id"]) for item in ranking_items]


def evaluate_matched_query(
    matched_query: dict[str, object],
    k_values: list[int],
    primary_k: int,
) -> dict[str, object]:
    """Evaluate before and after rankings for one validated query pair."""
    if primary_k not in k_values:
        raise ValueError("primary_k must be included in k_values")

    ground_truth = cast(dict[str, object], matched_query["ground_truth"])
    retrieval_result = cast(dict[str, object], matched_query["retrieval_result"])
    candidates = cast(list[dict[str, object]], retrieval_result["candidates"])

    chunk_to_document = {
        cast(str, candidate["chunk_id"]): cast(str, candidate["document_id"])
        for candidate in candidates
    }
    relevant_document_ids = set(
        cast(list[str], ground_truth["relevant_document_ids"])
    )

    before_ranked_document_ids = build_ranked_document_ids(
        _ranking_chunk_ids(retrieval_result["before_reranking"]),
        chunk_to_document,
    )
    after_ranked_document_ids = build_ranked_document_ids(
        _ranking_chunk_ids(retrieval_result["after_reranking"]),
        chunk_to_document,
    )

    before_metrics = evaluate_ranking(
        before_ranked_document_ids,
        relevant_document_ids,
        k_values,
    )
    after_metrics = evaluate_ranking(
        after_ranked_document_ids,
        relevant_document_ids,
        k_values,
    )

    depth_coverage = before_metrics.pop("depth_coverage_at_k")
    after_depth_coverage = after_metrics.pop("depth_coverage_at_k")
    if depth_coverage != after_depth_coverage:
        raise ValueError("before and after rankings must use the same candidate pool")

    metric_deltas = calculate_query_metric_deltas(
        cast(dict[str, object], before_metrics),
        cast(dict[str, object], after_metrics),
        k_values,
    )

    query_result: dict[str, object] = {
        "query_id": matched_query["query_id"],
        "query_text": ground_truth["query_text"],
        "relevant_document_ids": ground_truth["relevant_document_ids"],
        "before_reranking": {
            "ranked_document_ids": before_ranked_document_ids,
            "metrics": before_metrics,
        },
        "after_reranking": {
            "ranked_document_ids": after_ranked_document_ids,
            "metrics": after_metrics,
        },
        "metric_deltas": metric_deltas,
        "depth_coverage_at_k": depth_coverage,
    }
    query_result["failure_tags"] = generate_failure_tags(query_result, primary_k)
    return query_result


def _metrics_for_aggregation(
    query_result: dict[str, object],
    ranking_name: str,
) -> dict[str, dict[str, int | float] | float]:
    ranking_result = cast(dict[str, object], query_result[ranking_name])
    metrics = dict(
        cast(
            dict[str, dict[str, int | float] | float],
            ranking_result["metrics"],
        )
    )
    metrics["depth_coverage_at_k"] = cast(
        dict[str, int | float],
        query_result["depth_coverage_at_k"],
    )
    return metrics


def evaluate_matched_queries(
    matched_queries: list[dict[str, object]],
    k_values: list[int],
    primary_k: int,
) -> dict[str, object]:
    """Evaluate and aggregate every matched query before and after reranking."""
    if not matched_queries:
        raise ValueError("matched_queries must not be empty")

    query_results = [
        evaluate_matched_query(matched_query, k_values, primary_k)
        for matched_query in matched_queries
    ]
    before_query_metrics = [
        _metrics_for_aggregation(query_result, "before_reranking")
        for query_result in query_results
    ]
    after_query_metrics = [
        _metrics_for_aggregation(query_result, "after_reranking")
        for query_result in query_results
    ]

    before_aggregate = aggregate_metrics(before_query_metrics, k_values)
    after_aggregate = aggregate_metrics(after_query_metrics, k_values)
    depth_coverage = before_aggregate.pop("macro_depth_coverage_at_k")
    after_depth_coverage = after_aggregate.pop("macro_depth_coverage_at_k")
    if depth_coverage != after_depth_coverage:
        raise ValueError("before and after aggregates must use the same candidate pools")

    metric_deltas = calculate_aggregate_metric_deltas(
        cast(dict[str, object], before_aggregate),
        cast(dict[str, object], after_aggregate),
        k_values,
    )
    failure_analysis = summarize_failure_tags(query_results, primary_k)

    return {
        "primary_k": primary_k,
        "evaluated_query_count": len(query_results),
        "query_results": query_results,
        "failure_analysis": failure_analysis,
        "aggregate_metrics": {
            "before_reranking": before_aggregate,
            "after_reranking": after_aggregate,
            "metric_deltas": metric_deltas,
            "macro_depth_coverage_at_k": depth_coverage,
        },
    }
