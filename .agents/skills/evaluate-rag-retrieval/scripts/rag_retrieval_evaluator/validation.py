"""Validation for the evaluator's JSON-compatible input structures."""

from math import isfinite

SUPPORTED_SCHEMA_VERSION = "1.0"


class GroundTruthValidationError(ValueError):
    """Report every structural problem found in one ground truth dataset."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Ground truth validation failed:\n- " + "\n- ".join(errors))


class RetrievalResultsValidationError(ValueError):
    """Report every structural problem found in one retrieval result dataset."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Retrieval results validation failed:\n- " + "\n- ".join(errors))


def _is_non_blank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_ground_truth(data: object) -> dict[str, object]:
    """Validate and return a ground truth object using schema version 1.0."""
    if not isinstance(data, dict):
        raise GroundTruthValidationError(["ground_truth must be a JSON object"])

    errors: list[str] = []

    if data.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append("schema_version must be '1.0'")

    if "dataset_name" in data and not _is_non_blank_string(data["dataset_name"]):
        errors.append("dataset_name must be a non-blank string when provided")

    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        errors.append("queries must be a non-empty array")
    else:
        seen_query_ids: set[str] = set()

        for query_index, query in enumerate(queries):
            query_path = f"queries[{query_index}]"
            if not isinstance(query, dict):
                errors.append(f"{query_path} must be an object")
                continue

            query_id = query.get("query_id")
            if not _is_non_blank_string(query_id):
                errors.append(f"{query_path}.query_id must be a non-blank string")
            elif query_id in seen_query_ids:
                errors.append(f"{query_path}.query_id duplicates {query_id!r}")
            else:
                seen_query_ids.add(query_id)

            if not _is_non_blank_string(query.get("query_text")):
                errors.append(f"{query_path}.query_text must be a non-blank string")

            relevant_document_ids = query.get("relevant_document_ids")
            if not isinstance(relevant_document_ids, list) or not relevant_document_ids:
                errors.append(
                    f"{query_path}.relevant_document_ids must be a non-empty array"
                )
            else:
                seen_document_ids: set[str] = set()
                for document_index, document_id in enumerate(relevant_document_ids):
                    document_path = (
                        f"{query_path}.relevant_document_ids[{document_index}]"
                    )
                    if not _is_non_blank_string(document_id):
                        errors.append(f"{document_path} must be a non-blank string")
                    elif document_id in seen_document_ids:
                        errors.append(f"{document_path} duplicates {document_id!r}")
                    else:
                        seen_document_ids.add(document_id)

            if "metadata" in query and not isinstance(query["metadata"], dict):
                errors.append(f"{query_path}.metadata must be an object when provided")

    if errors:
        raise GroundTruthValidationError(errors)

    return data


def _validate_candidates(
    value: object,
    result_path: str,
    errors: list[str],
) -> tuple[set[str], bool]:
    candidate_path = f"{result_path}.candidates"
    if not isinstance(value, list):
        errors.append(f"{candidate_path} must be an array")
        return set(), False

    chunk_ids: set[str] = set()
    valid = True

    for candidate_index, candidate in enumerate(value):
        item_path = f"{candidate_path}[{candidate_index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{item_path} must be an object")
            valid = False
            continue

        chunk_id = candidate.get("chunk_id")
        if not _is_non_blank_string(chunk_id):
            errors.append(f"{item_path}.chunk_id must be a non-blank string")
            valid = False
        elif chunk_id in chunk_ids:
            errors.append(f"{item_path}.chunk_id duplicates {chunk_id!r}")
            valid = False
        else:
            chunk_ids.add(chunk_id)

        if not _is_non_blank_string(candidate.get("document_id")):
            errors.append(f"{item_path}.document_id must be a non-blank string")
            valid = False

        if "chunk_text" in candidate and not isinstance(candidate["chunk_text"], str):
            errors.append(f"{item_path}.chunk_text must be a string when provided")
            valid = False

        if "metadata" in candidate and not isinstance(candidate["metadata"], dict):
            errors.append(f"{item_path}.metadata must be an object when provided")
            valid = False

    return chunk_ids, valid


def _validate_ranking(
    value: object,
    result_path: str,
    ranking_name: str,
    errors: list[str],
    warnings: list[str],
) -> tuple[set[str], bool]:
    ranking_path = f"{result_path}.{ranking_name}"
    if not isinstance(value, list):
        errors.append(f"{ranking_path} must be an array")
        return set(), False

    chunk_ids: set[str] = set()
    valid = True

    for ranking_index, ranking_item in enumerate(value):
        item_path = f"{ranking_path}[{ranking_index}]"
        if not isinstance(ranking_item, dict):
            errors.append(f"{item_path} must be an object")
            valid = False
            continue

        chunk_id = ranking_item.get("chunk_id")
        if not _is_non_blank_string(chunk_id):
            errors.append(f"{item_path}.chunk_id must be a non-blank string")
            valid = False
        elif chunk_id in chunk_ids:
            errors.append(f"{item_path}.chunk_id duplicates {chunk_id!r}")
            valid = False
        else:
            chunk_ids.add(chunk_id)

        if "score" in ranking_item:
            score = ranking_item["score"]
            score_is_valid = (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and isfinite(score)
            )
            if not score_is_valid:
                warnings.append(
                    f"{item_path}.score should be a finite number; "
                    "array order remains authoritative"
                )

    return chunk_ids, valid


def validate_retrieval_results(
    data: object,
) -> tuple[dict[str, object], list[str]]:
    """Validate retrieval results and return the data plus non-fatal warnings."""
    if not isinstance(data, dict):
        raise RetrievalResultsValidationError(
            ["retrieval_results must be a JSON object"]
        )

    errors: list[str] = []
    warnings: list[str] = []

    if data.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        errors.append("schema_version must be '1.0'")

    if "run_name" in data and not _is_non_blank_string(data["run_name"]):
        errors.append("run_name must be a non-blank string when provided")

    results = data.get("results")
    if not isinstance(results, list):
        errors.append("results must be an array")
    else:
        seen_query_ids: set[str] = set()

        for result_index, result in enumerate(results):
            result_path = f"results[{result_index}]"
            if not isinstance(result, dict):
                errors.append(f"{result_path} must be an object")
                continue

            query_id = result.get("query_id")
            if not _is_non_blank_string(query_id):
                errors.append(f"{result_path}.query_id must be a non-blank string")
            elif query_id in seen_query_ids:
                errors.append(f"{result_path}.query_id duplicates {query_id!r}")
            else:
                seen_query_ids.add(query_id)

            candidate_ids, candidates_valid = _validate_candidates(
                result.get("candidates"),
                result_path,
                errors,
            )
            before_ids, before_valid = _validate_ranking(
                result.get("before_reranking"),
                result_path,
                "before_reranking",
                errors,
                warnings,
            )
            after_ids, after_valid = _validate_ranking(
                result.get("after_reranking"),
                result_path,
                "after_reranking",
                errors,
                warnings,
            )

            if candidates_valid and before_valid and before_ids != candidate_ids:
                errors.append(
                    f"{result_path}.before_reranking chunk_id set must exactly "
                    "match candidates"
                )
            if candidates_valid and after_valid and after_ids != candidate_ids:
                errors.append(
                    f"{result_path}.after_reranking chunk_id set must exactly "
                    "match candidates"
                )

    if errors:
        raise RetrievalResultsValidationError(errors)

    return data, warnings
