import unittest

from tests import _skill_path  # noqa: F401
from rag_retrieval_evaluator.comparison import (
    calculate_aggregate_metric_deltas,
    calculate_query_metric_deltas,
)


class CalculateQueryMetricDeltasTests(unittest.TestCase):
    def test_calculates_positive_zero_and_negative_deltas(self) -> None:
        before = {
            "hit_at_k": {"1": 0, "5": 1},
            "precision_at_k": {"1": 0.0, "5": 0.4},
            "recall_at_k": {"1": 0.0, "5": 1.0},
            "reciprocal_rank": 0.5,
            "ndcg_at_k": {"1": 0.0, "5": 0.8},
        }
        after = {
            "hit_at_k": {"1": 1, "5": 1},
            "precision_at_k": {"1": 1.0, "5": 0.2},
            "recall_at_k": {"1": 0.5, "5": 0.5},
            "reciprocal_rank": 1.0,
            "ndcg_at_k": {"1": 1.0, "5": 0.7},
        }

        result = calculate_query_metric_deltas(before, after, [1, 5])

        self.assertEqual(result["hit_at_k"], {"1": 1.0, "5": 0.0})
        self.assertEqual(result["precision_at_k"]["1"], 1.0)
        self.assertAlmostEqual(result["precision_at_k"]["5"], -0.2)
        self.assertEqual(result["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_k"]["5"], -0.1)


class CalculateAggregateMetricDeltasTests(unittest.TestCase):
    def test_calculates_dataset_level_deltas(self) -> None:
        before = {
            "hit_rate_at_k": {"1": 0.5},
            "macro_precision_at_k": {"1": 0.5},
            "macro_recall_at_k": {"1": 0.5},
            "mrr": 0.75,
            "macro_ndcg_at_k": {"1": 0.5},
        }
        after = {
            "hit_rate_at_k": {"1": 1.0},
            "macro_precision_at_k": {"1": 1.0},
            "macro_recall_at_k": {"1": 1.0},
            "mrr": 1.0,
            "macro_ndcg_at_k": {"1": 1.0},
        }

        result = calculate_aggregate_metric_deltas(before, after, [1])

        self.assertEqual(result["hit_rate_at_k"]["1"], 0.5)
        self.assertEqual(result["macro_precision_at_k"]["1"], 0.5)
        self.assertEqual(result["macro_recall_at_k"]["1"], 0.5)
        self.assertEqual(result["mrr"], 0.25)
        self.assertEqual(result["macro_ndcg_at_k"]["1"], 0.5)


if __name__ == "__main__":
    unittest.main()
