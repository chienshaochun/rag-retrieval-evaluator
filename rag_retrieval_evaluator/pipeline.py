"""Run the complete in-memory RAG retrieval evaluation workflow."""

from typing import cast

from .evaluation import evaluate_matched_queries
from .pairing import pair_query_records
from .validation import validate_ground_truth, validate_retrieval_results

OUTPUT_SCHEMA_VERSION = "1.0"


def _validate_evaluation_configuration(
    k_values: list[int],
    primary_k: int,
) -> None:
    if not k_values:
        raise ValueError("k_values must not be empty")
    if any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0
        for k in k_values
    ):
        raise ValueError("k_values must contain only positive integers")
    if len(k_values) != len(set(k_values)):
        raise ValueError("k_values must not contain duplicates")
    if not isinstance(primary_k, int) or isinstance(primary_k, bool) or primary_k <= 0:
        raise ValueError("primary_k must be a positive integer")
    if primary_k not in k_values:
        raise ValueError("primary_k must be included in k_values")


def run_evaluation_pipeline(
    ground_truth_data: object,
    retrieval_results_data: object,
    k_values: list[int],
    primary_k: int,
) -> dict[str, object]:
    """Validate, pair, evaluate, and assemble one structured result."""
    _validate_evaluation_configuration(k_values, primary_k)
    ground_truth = validate_ground_truth(ground_truth_data)
    retrieval_results, validation_warnings = validate_retrieval_results(
        retrieval_results_data
    )
    pairing_result = pair_query_records(ground_truth, retrieval_results)
    matched_queries = cast(
        list[dict[str, object]], pairing_result["matched_queries"]
    )

    pairing_summary = {
        "missing_retrieval_result_query_ids": pairing_result[
            "missing_retrieval_result_query_ids"
        ],
        "missing_ground_truth_query_ids": pairing_result[
            "missing_ground_truth_query_ids"
        ],
        "coverage": pairing_result["coverage"],
    }

    evaluation = (
        evaluate_matched_queries(matched_queries, k_values, primary_k)
        if matched_queries
        else None
    )

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "completed" if evaluation is not None else "no_matched_queries",
        "dataset_name": ground_truth.get("dataset_name"),
        "run_name": retrieval_results.get("run_name"),
        "configuration": {
            "k_values": list(k_values),
            "primary_k": primary_k,
        },
        "validation_warnings": validation_warnings,
        "pairing": pairing_summary,
        "evaluation": evaluation,
    }
