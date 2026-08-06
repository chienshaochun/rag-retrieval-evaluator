"""Pair validated ground truth queries with validated retrieval results."""


def _get_validated_records(
    dataset: dict[str, object],
    field_name: str,
) -> list[dict[str, object]]:
    records = dataset.get(field_name)
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(
            f"{field_name} must be validated before query records are paired"
        )
    return records


def pair_query_records(
    ground_truth: dict[str, object],
    retrieval_results: dict[str, object],
) -> dict[str, object]:
    """Pair records by exact query_id and report both coverage directions."""
    ground_truth_queries = _get_validated_records(ground_truth, "queries")
    retrieval_query_results = _get_validated_records(retrieval_results, "results")

    if not ground_truth_queries:
        raise ValueError("ground truth must contain at least one validated query")

    ground_truth_by_id = {
        query["query_id"]: query for query in ground_truth_queries
    }
    retrieval_result_by_id = {
        result["query_id"]: result for result in retrieval_query_results
    }

    matched_queries: list[dict[str, object]] = []
    missing_retrieval_result_query_ids: list[object] = []

    for query in ground_truth_queries:
        query_id = query["query_id"]
        retrieval_result = retrieval_result_by_id.get(query_id)
        if retrieval_result is None:
            missing_retrieval_result_query_ids.append(query_id)
            continue

        matched_queries.append(
            {
                "query_id": query_id,
                "ground_truth": query,
                "retrieval_result": retrieval_result,
            }
        )

    missing_ground_truth_query_ids = [
        result["query_id"]
        for result in retrieval_query_results
        if result["query_id"] not in ground_truth_by_id
    ]

    matched_query_count = len(matched_queries)
    ground_truth_query_count = len(ground_truth_queries)
    retrieval_result_query_count = len(retrieval_query_results)

    return {
        "matched_queries": matched_queries,
        "missing_retrieval_result_query_ids": missing_retrieval_result_query_ids,
        "missing_ground_truth_query_ids": missing_ground_truth_query_ids,
        "coverage": {
            "ground_truth_query_count": ground_truth_query_count,
            "retrieval_result_query_count": retrieval_result_query_count,
            "matched_query_count": matched_query_count,
            "result_coverage": matched_query_count / ground_truth_query_count,
            "ground_truth_coverage": (
                matched_query_count / retrieval_result_query_count
                if retrieval_result_query_count
                else None
            ),
        },
    }
