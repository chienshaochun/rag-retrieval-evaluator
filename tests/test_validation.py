import unittest

from rag_retrieval_evaluator.validation import (
    GroundTruthValidationError,
    validate_ground_truth,
)


class ValidateGroundTruthTests(unittest.TestCase):
    def test_accepts_complete_valid_ground_truth(self) -> None:
        data = {
            "schema_version": "1.0",
            "dataset_name": "maintenance-qa",
            "queries": [
                {
                    "query_id": "query-001",
                    "query_text": "How do I reset the pump alarm?",
                    "relevant_document_ids": ["doc-A", "doc-B"],
                    "metadata": {"category": "pump"},
                }
            ],
        }

        result = validate_ground_truth(data)

        self.assertIs(result, data)

    def test_accepts_ground_truth_without_optional_fields(self) -> None:
        data = {
            "schema_version": "1.0",
            "queries": [
                {
                    "query_id": "query-001",
                    "query_text": "What caused the alarm?",
                    "relevant_document_ids": ["doc-A"],
                }
            ],
        }

        self.assertIs(validate_ground_truth(data), data)

    def test_rejects_non_object_root(self) -> None:
        with self.assertRaises(GroundTruthValidationError) as context:
            validate_ground_truth([])

        self.assertEqual(context.exception.errors, ["ground_truth must be a JSON object"])

    def test_rejects_missing_or_unsupported_schema_version(self) -> None:
        for schema_version in (None, "2.0"):
            with self.subTest(schema_version=schema_version):
                data = {
                    "queries": [
                        {
                            "query_id": "query-001",
                            "query_text": "Question",
                            "relevant_document_ids": ["doc-A"],
                        }
                    ]
                }
                if schema_version is not None:
                    data["schema_version"] = schema_version

                with self.assertRaisesRegex(
                    GroundTruthValidationError,
                    "schema_version must be '1.0'",
                ):
                    validate_ground_truth(data)

    def test_rejects_missing_or_empty_queries(self) -> None:
        for queries in (None, [], "not-an-array"):
            with self.subTest(queries=queries):
                data = {"schema_version": "1.0"}
                if queries is not None:
                    data["queries"] = queries

                with self.assertRaisesRegex(
                    GroundTruthValidationError,
                    "queries must be a non-empty array",
                ):
                    validate_ground_truth(data)

    def test_collects_invalid_query_field_errors(self) -> None:
        data = {
            "schema_version": "1.0",
            "queries": [
                {
                    "query_id": " ",
                    "query_text": "",
                    "relevant_document_ids": ["", 123],
                    "metadata": "not-an-object",
                }
            ],
        }

        with self.assertRaises(GroundTruthValidationError) as context:
            validate_ground_truth(data)

        self.assertEqual(
            context.exception.errors,
            [
                "queries[0].query_id must be a non-blank string",
                "queries[0].query_text must be a non-blank string",
                "queries[0].relevant_document_ids[0] must be a non-blank string",
                "queries[0].relevant_document_ids[1] must be a non-blank string",
                "queries[0].metadata must be an object when provided",
            ],
        )

    def test_rejects_duplicate_query_ids(self) -> None:
        data = {
            "schema_version": "1.0",
            "queries": [
                {
                    "query_id": "query-001",
                    "query_text": "First question",
                    "relevant_document_ids": ["doc-A"],
                },
                {
                    "query_id": "query-001",
                    "query_text": "Second question",
                    "relevant_document_ids": ["doc-B"],
                },
            ],
        }

        with self.assertRaisesRegex(
            GroundTruthValidationError,
            "queries\\[1\\]\\.query_id duplicates 'query-001'",
        ):
            validate_ground_truth(data)

    def test_rejects_duplicate_relevant_document_ids(self) -> None:
        data = {
            "schema_version": "1.0",
            "queries": [
                {
                    "query_id": "query-001",
                    "query_text": "Question",
                    "relevant_document_ids": ["doc-A", "doc-A"],
                }
            ],
        }

        with self.assertRaisesRegex(
            GroundTruthValidationError,
            "relevant_document_ids\\[1\\] duplicates 'doc-A'",
        ):
            validate_ground_truth(data)

    def test_rejects_invalid_optional_dataset_name(self) -> None:
        data = {
            "schema_version": "1.0",
            "dataset_name": " ",
            "queries": [
                {
                    "query_id": "query-001",
                    "query_text": "Question",
                    "relevant_document_ids": ["doc-A"],
                }
            ],
        }

        with self.assertRaisesRegex(
            GroundTruthValidationError,
            "dataset_name must be a non-blank string when provided",
        ):
            validate_ground_truth(data)


if __name__ == "__main__":
    unittest.main()
