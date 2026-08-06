import unittest

from rag_retrieval_evaluator.pipeline import run_evaluation_pipeline
from rag_retrieval_evaluator.validation import GroundTruthValidationError


def ground_truth_query(query_id: str, document_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_text": f"Question for {query_id}",
        "relevant_document_ids": [document_id],
    }


def retrieval_result(
    query_id: str,
    document_id: str,
    score: object = 1.0,
) -> dict[str, object]:
    chunk_id = f"chunk-{query_id}"
    return {
        "query_id": query_id,
        "candidates": [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
            }
        ],
        "before_reranking": [{"chunk_id": chunk_id, "score": score}],
        "after_reranking": [{"chunk_id": chunk_id, "score": score}],
    }


class RunEvaluationPipelineTests(unittest.TestCase):
    def test_runs_complete_workflow_and_preserves_pairing_information(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "dataset_name": "maintenance-qa",
            "queries": [
                ground_truth_query("query-001", "doc-A"),
                ground_truth_query("query-002", "doc-B"),
            ],
        }
        retrieval_results = {
            "schema_version": "1.0",
            "run_name": "reranker-v1",
            "results": [
                retrieval_result("query-001", "doc-A", score="invalid"),
                retrieval_result("query-003", "doc-C"),
            ],
        }

        result = run_evaluation_pipeline(
            ground_truth,
            retrieval_results,
            k_values=[1, 5],
            primary_k=1,
        )

        self.assertEqual(result["schema_version"], "1.0")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["dataset_name"], "maintenance-qa")
        self.assertEqual(result["run_name"], "reranker-v1")
        self.assertEqual(
            result["configuration"],
            {"k_values": [1, 5], "primary_k": 1},
        )
        self.assertEqual(len(result["validation_warnings"]), 2)
        self.assertEqual(
            result["pairing"]["missing_retrieval_result_query_ids"],
            ["query-002"],
        )
        self.assertEqual(
            result["pairing"]["missing_ground_truth_query_ids"],
            ["query-003"],
        )
        self.assertEqual(result["pairing"]["coverage"]["result_coverage"], 0.5)
        self.assertEqual(result["evaluation"]["evaluated_query_count"], 1)
        self.assertEqual(
            result["evaluation"]["query_results"][0]["query_id"],
            "query-001",
        )

    def test_returns_no_matched_status_instead_of_inventing_metrics(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001", "doc-A")],
        }
        retrieval_results = {
            "schema_version": "1.0",
            "results": [retrieval_result("query-002", "doc-B")],
        }

        result = run_evaluation_pipeline(
            ground_truth,
            retrieval_results,
            k_values=[1],
            primary_k=1,
        )

        self.assertEqual(result["status"], "no_matched_queries")
        self.assertIsNone(result["evaluation"])
        self.assertEqual(result["pairing"]["coverage"]["matched_query_count"], 0)

    def test_rejects_invalid_configuration_even_when_no_queries_match(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001", "doc-A")],
        }
        retrieval_results = {"schema_version": "1.0", "results": []}

        with self.assertRaisesRegex(
            ValueError,
            "primary_k must be included in k_values",
        ):
            run_evaluation_pipeline(
                ground_truth,
                retrieval_results,
                k_values=[1, 5],
                primary_k=3,
            )

    def test_stops_before_pairing_when_ground_truth_is_invalid(self) -> None:
        with self.assertRaises(GroundTruthValidationError):
            run_evaluation_pipeline(
                {"schema_version": "1.0", "queries": []},
                {"schema_version": "1.0", "results": []},
                k_values=[1],
                primary_k=1,
            )


if __name__ == "__main__":
    unittest.main()
