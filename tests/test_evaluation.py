import unittest

from rag_retrieval_evaluator.evaluation import evaluate_matched_query


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

        result = evaluate_matched_query(query_pair, k_values=[1, 2])

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

        result = evaluate_matched_query(query_pair, k_values=[1, 2])

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

        result = evaluate_matched_query(query_pair, k_values=[1, 2, 5])

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

        result = evaluate_matched_query(query_pair, k_values=[1, 5])

        self.assertEqual(
            result["before_reranking"]["metrics"]["hit_at_k"],
            {"1": 0, "5": 0},
        )
        self.assertEqual(
            result["after_reranking"]["metrics"]["hit_at_k"],
            {"1": 0, "5": 0},
        )
        self.assertEqual(result["depth_coverage_at_k"], {"1": 0.0, "5": 0.0})


if __name__ == "__main__":
    unittest.main()
