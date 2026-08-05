import unittest

from rag_retrieval_evaluator.metrics import (
    aggregate_metrics,
    build_ranked_document_ids,
    depth_coverage_at_k,
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class BuildRankedDocumentIdsTests(unittest.TestCase):
    def test_converts_chunks_to_documents_in_ranking_order(self) -> None:
        result = build_ranked_document_ids(
            ranked_chunk_ids=["chunk-A1", "chunk-B1", "chunk-C1"],
            chunk_to_document={
                "chunk-A1": "doc-A",
                "chunk-B1": "doc-B",
                "chunk-C1": "doc-C",
            },
        )

        self.assertEqual(result, ["doc-A", "doc-B", "doc-C"])

    def test_keeps_only_first_occurrence_of_each_document(self) -> None:
        result = build_ranked_document_ids(
            ranked_chunk_ids=["chunk-A1", "chunk-A2", "chunk-B1", "chunk-A3"],
            chunk_to_document={
                "chunk-A1": "doc-A",
                "chunk-A2": "doc-A",
                "chunk-A3": "doc-A",
                "chunk-B1": "doc-B",
            },
        )

        self.assertEqual(result, ["doc-A", "doc-B"])

    def test_returns_empty_ranking_for_empty_input(self) -> None:
        result = build_ranked_document_ids([], {})

        self.assertEqual(result, [])

    def test_rejects_chunk_that_is_missing_from_candidate_mapping(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown chunk_id in ranking: 'chunk-missing'",
        ):
            build_ranked_document_ids(
                ranked_chunk_ids=["chunk-known", "chunk-missing"],
                chunk_to_document={"chunk-known": "doc-A"},
            )


class HitAtKTests(unittest.TestCase):
    def test_returns_one_when_relevant_document_is_in_top_k(self) -> None:
        result = hit_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-B"},
            k=2,
        )

        self.assertEqual(result, 1)

    def test_returns_zero_when_relevant_document_is_below_top_k(self) -> None:
        result = hit_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-C"},
            k=2,
        )

        self.assertEqual(result, 0)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = hit_at_k(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
            k=5,
        )

        self.assertEqual(result, 0)

    def test_any_relevant_document_is_enough_for_a_hit(self) -> None:
        result = hit_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-B", "doc-D"},
            k=3,
        )

        self.assertEqual(result, 1)

    def test_rejects_non_positive_k(self) -> None:
        for invalid_k in (0, -1):
            with self.subTest(k=invalid_k):
                with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
                    hit_at_k(
                        ranked_document_ids=["doc-A"],
                        relevant_document_ids={"doc-A"},
                        k=invalid_k,
                    )


class PrecisionAtKTests(unittest.TestCase):
    def test_divides_relevant_documents_in_top_k_by_k(self) -> None:
        result = precision_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-A", "doc-C"},
            k=3,
        )

        self.assertAlmostEqual(result, 2 / 3)

    def test_ignores_relevant_documents_below_top_k(self) -> None:
        result = precision_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-C"},
            k=2,
        )

        self.assertEqual(result, 0.0)

    def test_uses_k_as_denominator_when_ranking_is_shallow(self) -> None:
        result = precision_at_k(
            ranked_document_ids=["doc-A", "doc-B"],
            relevant_document_ids={"doc-A"},
            k=5,
        )

        self.assertEqual(result, 0.2)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = precision_at_k(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
            k=5,
        )

        self.assertEqual(result, 0.0)

    def test_rejects_non_positive_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            precision_at_k(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k=0,
            )


class RecallAtKTests(unittest.TestCase):
    def test_divides_relevant_documents_in_top_k_by_all_relevant_documents(self) -> None:
        result = recall_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-A", "doc-C", "doc-D"},
            k=3,
        )

        self.assertAlmostEqual(result, 2 / 3)

    def test_ignores_relevant_documents_below_top_k(self) -> None:
        result = recall_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-C"},
            k=2,
        )

        self.assertEqual(result, 0.0)

    def test_uses_all_ground_truth_documents_as_denominator(self) -> None:
        result = recall_at_k(
            ranked_document_ids=["doc-A", "doc-B"],
            relevant_document_ids={"doc-A", "doc-C", "doc-D"},
            k=5,
        )

        self.assertAlmostEqual(result, 1 / 3)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = recall_at_k(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
            k=5,
        )

        self.assertEqual(result, 0.0)

    def test_rejects_empty_relevant_document_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "relevant_document_ids must not be empty",
        ):
            recall_at_k(
                ranked_document_ids=["doc-A"],
                relevant_document_ids=set(),
                k=1,
            )

    def test_rejects_non_positive_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            recall_at_k(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k=0,
            )


class ReciprocalRankTests(unittest.TestCase):
    def test_returns_one_when_first_document_is_relevant(self) -> None:
        result = reciprocal_rank(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-A"},
        )

        self.assertEqual(result, 1.0)

    def test_returns_inverse_position_of_first_relevant_document(self) -> None:
        result = reciprocal_rank(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-C"},
        )

        self.assertAlmostEqual(result, 1 / 3)

    def test_uses_only_first_relevant_document(self) -> None:
        result = reciprocal_rank(
            ranked_document_ids=["doc-A", "doc-B", "doc-C", "doc-D"],
            relevant_document_ids={"doc-B", "doc-D"},
        )

        self.assertEqual(result, 0.5)

    def test_returns_zero_when_no_relevant_document_is_retrieved(self) -> None:
        result = reciprocal_rank(
            ranked_document_ids=["doc-A", "doc-B"],
            relevant_document_ids={"doc-C"},
        )

        self.assertEqual(result, 0.0)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = reciprocal_rank(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
        )

        self.assertEqual(result, 0.0)

    def test_rejects_empty_relevant_document_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "relevant_document_ids must not be empty",
        ):
            reciprocal_rank(
                ranked_document_ids=["doc-A"],
                relevant_document_ids=set(),
            )


class NdcgAtKTests(unittest.TestCase):
    def test_returns_one_for_ideal_ranking(self) -> None:
        result = ndcg_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C"],
            relevant_document_ids={"doc-A", "doc-B"},
            k=3,
        )

        self.assertEqual(result, 1.0)

    def test_discounts_relevant_documents_at_lower_ranks(self) -> None:
        result = ndcg_at_k(
            ranked_document_ids=["doc-X", "doc-A", "doc-B"],
            relevant_document_ids={"doc-A", "doc-B"},
            k=3,
        )

        self.assertAlmostEqual(result, 0.6934264036)

    def test_ignores_relevant_documents_below_top_k(self) -> None:
        result = ndcg_at_k(
            ranked_document_ids=["doc-X", "doc-A"],
            relevant_document_ids={"doc-A"},
            k=1,
        )

        self.assertEqual(result, 0.0)

    def test_ideal_ranking_uses_all_possible_relevant_positions(self) -> None:
        result = ndcg_at_k(
            ranked_document_ids=["doc-A"],
            relevant_document_ids={"doc-A", "doc-B"},
            k=5,
        )

        self.assertAlmostEqual(result, 0.6131471928)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = ndcg_at_k(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
            k=5,
        )

        self.assertEqual(result, 0.0)

    def test_rejects_empty_relevant_document_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "relevant_document_ids must not be empty",
        ):
            ndcg_at_k(
                ranked_document_ids=["doc-A"],
                relevant_document_ids=set(),
                k=1,
            )

    def test_rejects_non_positive_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            ndcg_at_k(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k=0,
            )


class DepthCoverageAtKTests(unittest.TestCase):
    def test_returns_one_when_at_least_k_documents_are_available(self) -> None:
        result = depth_coverage_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C", "doc-D", "doc-E"],
            k=5,
        )

        self.assertEqual(result, 1.0)

    def test_returns_fraction_when_fewer_than_k_documents_are_available(self) -> None:
        result = depth_coverage_at_k(
            ranked_document_ids=["doc-A", "doc-B"],
            k=5,
        )

        self.assertEqual(result, 0.4)

    def test_counts_each_document_only_once(self) -> None:
        result = depth_coverage_at_k(
            ranked_document_ids=["doc-A", "doc-A", "doc-B"],
            k=5,
        )

        self.assertEqual(result, 0.4)

    def test_caps_coverage_at_one(self) -> None:
        result = depth_coverage_at_k(
            ranked_document_ids=["doc-A", "doc-B", "doc-C", "doc-D"],
            k=2,
        )

        self.assertEqual(result, 1.0)

    def test_returns_zero_for_empty_ranking(self) -> None:
        result = depth_coverage_at_k(
            ranked_document_ids=[],
            k=5,
        )

        self.assertEqual(result, 0.0)

    def test_rejects_non_positive_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            depth_coverage_at_k(
                ranked_document_ids=["doc-A"],
                k=0,
            )


class EvaluateRankingTests(unittest.TestCase):
    def test_returns_all_metrics_for_each_requested_k(self) -> None:
        result = evaluate_ranking(
            ranked_document_ids=["doc-X", "doc-A", "doc-B"],
            relevant_document_ids={"doc-A", "doc-B"},
            k_values=[1, 3, 5],
        )

        self.assertEqual(result["hit_at_k"], {"1": 0, "3": 1, "5": 1})
        self.assertEqual(result["precision_at_k"]["1"], 0.0)
        self.assertAlmostEqual(result["precision_at_k"]["3"], 2 / 3)
        self.assertEqual(result["precision_at_k"]["5"], 0.4)
        self.assertEqual(result["recall_at_k"], {"1": 0.0, "3": 1.0, "5": 1.0})
        self.assertEqual(result["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_k"]["3"], 0.6934264036)
        self.assertEqual(
            result["depth_coverage_at_k"],
            {"1": 1.0, "3": 1.0, "5": 0.6},
        )

    def test_returns_zero_metrics_for_empty_ranking(self) -> None:
        result = evaluate_ranking(
            ranked_document_ids=[],
            relevant_document_ids={"doc-A"},
            k_values=[1, 5],
        )

        self.assertEqual(result["hit_at_k"], {"1": 0, "5": 0})
        self.assertEqual(result["precision_at_k"], {"1": 0.0, "5": 0.0})
        self.assertEqual(result["recall_at_k"], {"1": 0.0, "5": 0.0})
        self.assertEqual(result["reciprocal_rank"], 0.0)
        self.assertEqual(result["ndcg_at_k"], {"1": 0.0, "5": 0.0})
        self.assertEqual(result["depth_coverage_at_k"], {"1": 0.0, "5": 0.0})

    def test_rejects_empty_k_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_values must not be empty"):
            evaluate_ranking(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k_values=[],
            )

    def test_rejects_duplicate_k_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "k_values must not contain duplicates",
        ):
            evaluate_ranking(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k_values=[1, 5, 5],
            )

    def test_rejects_non_positive_k_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            evaluate_ranking(
                ranked_document_ids=["doc-A"],
                relevant_document_ids={"doc-A"},
                k_values=[1, 0],
            )

    def test_rejects_empty_relevant_document_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "relevant_document_ids must not be empty",
        ):
            evaluate_ranking(
                ranked_document_ids=["doc-A"],
                relevant_document_ids=set(),
                k_values=[1],
            )


class AggregateMetricsTests(unittest.TestCase):
    def test_macro_averages_all_query_metrics(self) -> None:
        query_metrics = [
            evaluate_ranking(
                ranked_document_ids=["doc-A", "doc-X"],
                relevant_document_ids={"doc-A"},
                k_values=[1, 2],
            ),
            evaluate_ranking(
                ranked_document_ids=["doc-X", "doc-B"],
                relevant_document_ids={"doc-B"},
                k_values=[1, 2],
            ),
        ]

        result = aggregate_metrics(query_metrics, k_values=[1, 2])

        self.assertEqual(result["hit_rate_at_k"], {"1": 0.5, "2": 1.0})
        self.assertEqual(result["macro_precision_at_k"], {"1": 0.5, "2": 0.5})
        self.assertEqual(result["macro_recall_at_k"], {"1": 0.5, "2": 1.0})
        self.assertEqual(result["mrr"], 0.75)
        self.assertEqual(result["macro_ndcg_at_k"]["1"], 0.5)
        self.assertAlmostEqual(result["macro_ndcg_at_k"]["2"], 0.8154648768)
        self.assertEqual(
            result["macro_depth_coverage_at_k"],
            {"1": 1.0, "2": 1.0},
        )

    def test_includes_zero_result_queries_in_average(self) -> None:
        query_metrics = [
            evaluate_ranking(["doc-A"], {"doc-A"}, [1]),
            evaluate_ranking([], {"doc-B"}, [1]),
        ]

        result = aggregate_metrics(query_metrics, k_values=[1])

        self.assertEqual(result["hit_rate_at_k"]["1"], 0.5)
        self.assertEqual(result["mrr"], 0.5)

    def test_rejects_empty_query_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_metrics must not be empty"):
            aggregate_metrics([], k_values=[1, 5])

    def test_rejects_missing_requested_k(self) -> None:
        query_metrics = [
            evaluate_ranking(["doc-A"], {"doc-A"}, [1]),
        ]

        with self.assertRaisesRegex(ValueError, "hit_at_k is missing k=5"):
            aggregate_metrics(query_metrics, k_values=[1, 5])


if __name__ == "__main__":
    unittest.main()
