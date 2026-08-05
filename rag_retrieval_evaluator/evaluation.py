"""Evaluate validated and paired retrieval records."""

from typing import cast

from .metrics import build_ranked_document_ids, evaluate_ranking


def _ranking_chunk_ids(ranking: object) -> list[str]:
    ranking_items = cast(list[dict[str, object]], ranking)
    return [cast(str, item["chunk_id"]) for item in ranking_items]


def evaluate_matched_query(
    matched_query: dict[str, object],
    k_values: list[int],
) -> dict[str, object]:
    """Evaluate before and after rankings for one validated query pair."""
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

    return {
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
        "depth_coverage_at_k": depth_coverage,
    }
