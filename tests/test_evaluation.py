import unittest

from rag_retrieval_evaluator.evaluation import (
    evaluate_matched_queries,
    evaluate_matched_query,
)


def matched_query(
    candidates: list[dict[str, object]],
    before_chunk_ids: list[str],
    after_chunk_ids: list[str],
    relevant_document_ids: list[str],
) -> dict[str, object]:
    return {
        "query_id": "query-001",
        "ground_truth": {
            "query_id": "query-001",
            "query_text": "How do I reset the pump alarm?",
            "relevant_document_ids": relevant_document_ids,
        },
        "retrieval_result": {
            "query_id": "query-001",
            "candidates": candidates,
            "before_reranking": [
                {"chunk_id": chunk_id} for chunk_id in before_chunk_ids
            ],
            "after_reranking": [
                {"chunk_id": chunk_id} for chunk_id in after_chunk_ids
            ],
        },
    }


class EvaluateMatchedQueryTests(unittest.TestCase):
    def test_evaluates_before_and_after_rankings_separately(self) -> None:
        query_pair = matched_query(
            candidates=[
                {"chunk_id": "chunk-X", "document_id": "doc-X"},
                {"chunk_id": "chunk-A", "document_id": "doc-A"},
            ],
            before_chunk_ids=["chunk-X", "chunk-A"],
            after_chunk_ids=["chunk-A", "chunk-X"],
            relevant_document_ids=["doc-A"],
        )

        result = evaluate_matched_query(query_pair, k_values=[1, 2], primary_k=1)

        before = result["before_reranking"]
        after = result["after_reranking"]
        self.assertEqual(before["ranked_document_ids"], ["doc-X", "doc-A"])
        self.assertEqual(after["ranked_document_ids"], ["doc-A", "doc-X"])
        self.assertEqual(before["metrics"]["hit_at_k"]["1"], 0)
        self.assertEqual(after["metrics"]["hit_at_k"]["1"], 1)
        self.assertEqual(before["metrics"]["reciprocal_rank"], 0.5)
        self.assertEqual(after["metrics"]["reciprocal_rank"], 1.0)
        self.assertEqual(before["metrics"]["ndcg_at_k"]["1"], 0.0)
        self.assertEqual(after["metrics"]["ndcg_at_k"]["1"], 1.0)
        self.assertEqual(result["metric_deltas"]["hit_at_k"]["1"], 1.0)
        self.assertEqual(result["metric_deltas"]["reciprocal_rank"], 0.5)
        self.assertEqual(result["metric_deltas"]["ndcg_at_k"]["1"], 1.0)
        self.assertEqual(
            result["failure_tags"],
            ["reranker_rescue", "reranker_improvement"],
        )

    def test_deduplicates_documents_before_calculating_metrics(self) -> None:
        query_pair = matched_query(
            candidates=[
                {"chunk_id": "chunk-A1", "document_id": "doc-A"},
                {"chunk_id": "chunk-A2", "document_id": "doc-A"},
                {"chunk_id": "chunk-B1", "document_id": "doc-B"},
            ],
            before_chunk_ids=["chunk-A1", "chunk-A2", "chunk-B1"],
            after_chunk_ids=["chunk-B1", "chunk-A2", "chunk-A1"],
            relevant_document_ids=["doc-B"],
        )

        result = evaluate_matched_query(query_pair, k_values=[1, 2], primary_k=1)

        self.assertEqual(
            result["before_reranking"]["ranked_document_ids"],
            ["doc-A", "doc-B"],
        )
        self.assertEqual(
            result["after_reranking"]["ranked_document_ids"],
            ["doc-B", "doc-A"],
        )

    def test_records_depth_coverage_once_for_the_shared_candidate_pool(self) -> None:
        query_pair = matched_query(
            candidates=[
                {"chunk_id": "chunk-A", "document_id": "doc-A"},
                {"chunk_id": "chunk-B", "document_id": "doc-B"},
            ],
            before_chunk_ids=["chunk-A", "chunk-B"],
            after_chunk_ids=["chunk-B", "chunk-A"],
            relevant_document_ids=["doc-A"],
        )

        result = evaluate_matched_query(
            query_pair,
            k_values=[1, 2, 5],
            primary_k=1,
        )

        self.assertEqual(
            result["depth_coverage_at_k"],
            {"1": 1.0, "2": 1.0, "5": 0.4},
        )
        self.assertNotIn("depth_coverage_at_k", result["before_reranking"]["metrics"])
        self.assertNotIn("depth_coverage_at_k", result["after_reranking"]["metrics"])

    def test_empty_retrieval_result_produces_zero_metrics(self) -> None:
        query_pair = matched_query(
            candidates=[],
            before_chunk_ids=[],
            after_chunk_ids=[],
            relevant_document_ids=["doc-A"],
        )

        result = evaluate_matched_query(query_pair, k_values=[1, 5], primary_k=1)

        self.assertEqual(
            result["before_reranking"]["metrics"]["hit_at_k"],
            {"1": 0, "5": 0},
        )
        self.assertEqual(
            result["after_reranking"]["metrics"]["hit_at_k"],
            {"1": 0, "5": 0},
        )
        self.assertEqual(result["depth_coverage_at_k"], {"1": 0.0, "5": 0.0})


class EvaluateMatchedQueriesTests(unittest.TestCase):
    def test_aggregates_before_and_after_metrics_separately(self) -> None:
        improved_query = matched_query(
            candidates=[
                {"chunk_id": "chunk-X", "document_id": "doc-X"},
                {"chunk_id": "chunk-A", "document_id": "doc-A"},
            ],
            before_chunk_ids=["chunk-X", "chunk-A"],
            after_chunk_ids=["chunk-A", "chunk-X"],
            relevant_document_ids=["doc-A"],
        )
        unchanged_query = {
            **matched_query(
                candidates=[
                    {"chunk_id": "chunk-B", "document_id": "doc-B"},
                    {"chunk_id": "chunk-Y", "document_id": "doc-Y"},
                ],
                before_chunk_ids=["chunk-B", "chunk-Y"],
                after_chunk_ids=["chunk-B", "chunk-Y"],
                relevant_document_ids=["doc-B"],
            ),
            "query_id": "query-002",
        }
        unchanged_query["ground_truth"]["query_id"] = "query-002"
        unchanged_query["ground_truth"]["query_text"] = "Question two"
        unchanged_query["retrieval_result"]["query_id"] = "query-002"

        result = evaluate_matched_queries(
            [improved_query, unchanged_query],
            k_values=[1, 2],
            primary_k=1,
        )

        aggregates = result["aggregate_metrics"]
        before = aggregates["before_reranking"]
        after = aggregates["after_reranking"]
        self.assertEqual(result["evaluated_query_count"], 2)
        self.assertEqual(before["hit_rate_at_k"], {"1": 0.5, "2": 1.0})
        self.assertEqual(after["hit_rate_at_k"], {"1": 1.0, "2": 1.0})
        self.assertEqual(before["mrr"], 0.75)
        self.assertEqual(after["mrr"], 1.0)
        self.assertAlmostEqual(before["macro_ndcg_at_k"]["2"], 0.8154648768)
        self.assertEqual(after["macro_ndcg_at_k"]["2"], 1.0)
        self.assertEqual(aggregates["metric_deltas"]["hit_rate_at_k"]["1"], 0.5)
        self.assertEqual(aggregates["metric_deltas"]["mrr"], 0.25)
        self.assertAlmostEqual(
            aggregates["metric_deltas"]["macro_ndcg_at_k"]["2"],
            0.1845351232,
        )
        self.assertEqual(result["failure_analysis"]["failure_case_count"], 0)

    def test_includes_explicit_zero_result_query_in_aggregates(self) -> None:
        successful_query = matched_query(
            candidates=[
                {"chunk_id": "chunk-A", "document_id": "doc-A"},
            ],
            before_chunk_ids=["chunk-A"],
            after_chunk_ids=["chunk-A"],
            relevant_document_ids=["doc-A"],
        )
        empty_query = {
            **matched_query(
                candidates=[],
                before_chunk_ids=[],
                after_chunk_ids=[],
                relevant_document_ids=["doc-B"],
            ),
            "query_id": "query-002",
        }
        empty_query["ground_truth"]["query_id"] = "query-002"
        empty_query["retrieval_result"]["query_id"] = "query-002"

        result = evaluate_matched_queries(
            [successful_query, empty_query],
            k_values=[1],
            primary_k=1,
        )

        aggregates = result["aggregate_metrics"]
        self.assertEqual(
            aggregates["before_reranking"]["hit_rate_at_k"]["1"],
            0.5,
        )
        self.assertEqual(
            aggregates["after_reranking"]["hit_rate_at_k"]["1"],
            0.5,
        )
        self.assertEqual(aggregates["macro_depth_coverage_at_k"]["1"], 0.5)
        failure_analysis = result["failure_analysis"]
        self.assertEqual(failure_analysis["failure_case_count"], 1)
        self.assertEqual(failure_analysis["failure_case_rate"], 0.5)
        self.assertEqual(
            failure_analysis["failure_cases"][0]["query_id"],
            "query-002",
        )

    def test_records_macro_depth_coverage_only_once(self) -> None:
        query_pair = matched_query(
            candidates=[
                {"chunk_id": "chunk-A", "document_id": "doc-A"},
            ],
            before_chunk_ids=["chunk-A"],
            after_chunk_ids=["chunk-A"],
            relevant_document_ids=["doc-A"],
        )

        result = evaluate_matched_queries(
            [query_pair],
            k_values=[1, 5],
            primary_k=1,
        )

        aggregates = result["aggregate_metrics"]
        self.assertEqual(
            aggregates["macro_depth_coverage_at_k"],
            {"1": 1.0, "5": 0.2},
        )
        self.assertNotIn(
            "macro_depth_coverage_at_k",
            aggregates["before_reranking"],
        )
        self.assertNotIn(
            "macro_depth_coverage_at_k",
            aggregates["after_reranking"],
        )

    def test_rejects_empty_matched_query_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched_queries must not be empty"):
            evaluate_matched_queries([], k_values=[1, 5], primary_k=1)

    def test_rejects_primary_k_not_in_k_values(self) -> None:
        query_pair = matched_query(
            candidates=[],
            before_chunk_ids=[],
            after_chunk_ids=[],
            relevant_document_ids=["doc-A"],
        )

        with self.assertRaisesRegex(
            ValueError,
            "primary_k must be included in k_values",
        ):
            evaluate_matched_query(query_pair, k_values=[1, 5], primary_k=3)


if __name__ == "__main__":
    unittest.main()
