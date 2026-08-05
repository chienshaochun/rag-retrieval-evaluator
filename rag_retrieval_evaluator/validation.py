"""Validation for the evaluator's JSON-compatible input structures."""

SUPPORTED_SCHEMA_VERSION = "1.0"


class GroundTruthValidationError(ValueError):
    """Report every structural problem found in one ground truth dataset."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Ground truth validation failed:\n- " + "\n- ".join(errors))


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
