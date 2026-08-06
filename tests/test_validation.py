import unittest

from tests import _skill_path  # noqa: F401
from rag_retrieval_evaluator.validation import (
    GroundTruthValidationError,
    RetrievalResultsValidationError,
    validate_ground_truth,
    validate_retrieval_results,
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


class ValidateRetrievalResultsTests(unittest.TestCase):
    def test_accepts_pure_reranking_of_the_same_candidates(self) -> None:
        data = {
            "schema_version": "1.0",
            "run_name": "baseline-vs-reranker",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [
                        {
                            "chunk_id": "chunk-A1",
                            "document_id": "doc-A",
                            "chunk_text": "Pump reset instructions",
                            "metadata": {"page": 1},
                        },
                        {
                            "chunk_id": "chunk-B1",
                            "document_id": "doc-B",
                        },
                    ],
                    "before_reranking": [
                        {"chunk_id": "chunk-B1", "score": 0.8},
                        {"chunk_id": "chunk-A1", "score": 0.7},
                    ],
                    "after_reranking": [
                        {"chunk_id": "chunk-A1", "score": 0.95},
                        {"chunk_id": "chunk-B1", "score": 0.2},
                    ],
                }
            ],
        }

        validated_data, warnings = validate_retrieval_results(data)

        self.assertIs(validated_data, data)
        self.assertEqual(warnings, [])

    def test_accepts_explicit_empty_retrieval_result(self) -> None:
        data = {
            "schema_version": "1.0",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [],
                    "before_reranking": [],
                    "after_reranking": [],
                }
            ],
        }

        _, warnings = validate_retrieval_results(data)

        self.assertEqual(warnings, [])

    def test_accepts_empty_results_array_for_later_coverage_handling(self) -> None:
        data = {"schema_version": "1.0", "results": []}

        self.assertEqual(validate_retrieval_results(data), (data, []))

    def test_rejects_non_object_root(self) -> None:
        with self.assertRaises(RetrievalResultsValidationError) as context:
            validate_retrieval_results([])

        self.assertEqual(
            context.exception.errors,
            ["retrieval_results must be a JSON object"],
        )

    def test_rejects_invalid_top_level_fields(self) -> None:
        data = {
            "schema_version": "2.0",
            "run_name": " ",
            "results": "not-an-array",
        }

        with self.assertRaises(RetrievalResultsValidationError) as context:
            validate_retrieval_results(data)

        self.assertEqual(
            context.exception.errors,
            [
                "schema_version must be '1.0'",
                "run_name must be a non-blank string when provided",
                "results must be an array",
            ],
        )

    def test_rejects_duplicate_result_query_ids(self) -> None:
        empty_result = {
            "query_id": "query-001",
            "candidates": [],
            "before_reranking": [],
            "after_reranking": [],
        }
        data = {
            "schema_version": "1.0",
            "results": [empty_result, dict(empty_result)],
        }

        with self.assertRaisesRegex(
            RetrievalResultsValidationError,
            "results\\[1\\]\\.query_id duplicates 'query-001'",
        ):
            validate_retrieval_results(data)

    def test_rejects_invalid_and_duplicate_candidates(self) -> None:
        data = {
            "schema_version": "1.0",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [
                        {"chunk_id": "chunk-A", "document_id": "doc-A"},
                        {
                            "chunk_id": "chunk-A",
                            "document_id": "",
                            "chunk_text": 123,
                            "metadata": "invalid",
                        },
                    ],
                    "before_reranking": [
                        {"chunk_id": "chunk-A"},
                    ],
                    "after_reranking": [
                        {"chunk_id": "chunk-A"},
                    ],
                }
            ],
        }

        with self.assertRaises(RetrievalResultsValidationError) as context:
            validate_retrieval_results(data)

        self.assertIn(
            "results[0].candidates[1].chunk_id duplicates 'chunk-A'",
            context.exception.errors,
        )
        self.assertIn(
            "results[0].candidates[1].document_id must be a non-blank string",
            context.exception.errors,
        )
        self.assertIn(
            "results[0].candidates[1].chunk_text must be a string when provided",
            context.exception.errors,
        )
        self.assertIn(
            "results[0].candidates[1].metadata must be an object when provided",
            context.exception.errors,
        )

    def test_rejects_duplicate_chunk_in_ranking(self) -> None:
        data = {
            "schema_version": "1.0",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [
                        {"chunk_id": "chunk-A", "document_id": "doc-A"},
                    ],
                    "before_reranking": [
                        {"chunk_id": "chunk-A"},
                        {"chunk_id": "chunk-A"},
                    ],
                    "after_reranking": [{"chunk_id": "chunk-A"}],
                }
            ],
        }

        with self.assertRaisesRegex(
            RetrievalResultsValidationError,
            "before_reranking\\[1\\]\\.chunk_id duplicates 'chunk-A'",
        ):
            validate_retrieval_results(data)

    def test_rejects_candidate_set_changes_before_or_after_reranking(self) -> None:
        data = {
            "schema_version": "1.0",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [
                        {"chunk_id": "chunk-A", "document_id": "doc-A"},
                        {"chunk_id": "chunk-B", "document_id": "doc-B"},
                    ],
                    "before_reranking": [{"chunk_id": "chunk-A"}],
                    "after_reranking": [
                        {"chunk_id": "chunk-A"},
                        {"chunk_id": "chunk-X"},
                    ],
                }
            ],
        }

        with self.assertRaises(RetrievalResultsValidationError) as context:
            validate_retrieval_results(data)

        self.assertIn(
            "results[0].before_reranking chunk_id set must exactly match candidates",
            context.exception.errors,
        )
        self.assertIn(
            "results[0].after_reranking chunk_id set must exactly match candidates",
            context.exception.errors,
        )

    def test_invalid_scores_create_warnings_instead_of_errors(self) -> None:
        data = {
            "schema_version": "1.0",
            "results": [
                {
                    "query_id": "query-001",
                    "candidates": [
                        {"chunk_id": "chunk-A", "document_id": "doc-A"},
                    ],
                    "before_reranking": [
                        {"chunk_id": "chunk-A", "score": "high"},
                    ],
                    "after_reranking": [
                        {"chunk_id": "chunk-A", "score": float("nan")},
                    ],
                }
            ],
        }

        validated_data, warnings = validate_retrieval_results(data)

        self.assertIs(validated_data, data)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all("array order remains authoritative" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
