import unittest

from rag_retrieval_evaluator.failure_analysis import (
    generate_failure_tags,
    summarize_failure_tags,
)


def query_result(
    relevant_document_ids: list[str],
    before_ranked_document_ids: list[str],
    after_ranked_document_ids: list[str],
    ndcg_delta: float,
    primary_k: int = 1,
) -> dict[str, object]:
    return {
        "relevant_document_ids": relevant_document_ids,
        "before_reranking": {
            "ranked_document_ids": before_ranked_document_ids,
        },
        "after_reranking": {
            "ranked_document_ids": after_ranked_document_ids,
        },
        "metric_deltas": {
            "ndcg_at_k": {str(primary_k): ndcg_delta},
        },
    }


class GenerateFailureTagsTests(unittest.TestCase):
    def test_tags_missing_candidate_and_persistent_miss(self) -> None:
        result = generate_failure_tags(
            query_result(["doc-A"], [], [], 0.0),
            primary_k=1,
        )

        self.assertEqual(
            result,
            [
                "no_relevant_candidate",
                "top_k_total_miss",
                "persistent_top_k_miss",
                "reranker_unchanged",
            ],
        )

    def test_tags_partial_coverage_rescue_and_improvement(self) -> None:
        result = generate_failure_tags(
            query_result(
                ["doc-A", "doc-B"],
                ["doc-X", "doc-A"],
                ["doc-A", "doc-X"],
                1.0,
            ),
            primary_k=1,
        )

        self.assertEqual(
            result,
            [
                "partial_relevant_candidate_coverage",
                "top_k_incomplete_coverage",
                "reranker_rescue",
                "reranker_improvement",
            ],
        )

    def test_tags_persistent_miss_when_relevant_candidate_stays_below_k(self) -> None:
        result = generate_failure_tags(
            query_result(
                ["doc-A"],
                ["doc-X", "doc-A"],
                ["doc-X", "doc-A"],
                0.0,
            ),
            primary_k=1,
        )

        self.assertEqual(
            result,
            [
                "top_k_total_miss",
                "persistent_top_k_miss",
                "reranker_unchanged",
            ],
        )

    def test_tags_regression_when_relevant_document_falls_out_of_top_k(self) -> None:
        result = generate_failure_tags(
            query_result(
                ["doc-A"],
                ["doc-A", "doc-X"],
                ["doc-X", "doc-A"],
                -1.0,
            ),
            primary_k=1,
        )

        self.assertEqual(
            result,
            ["top_k_total_miss", "reranker_regression"],
        )

    def test_tags_unchanged_successful_ranking(self) -> None:
        result = generate_failure_tags(
            query_result(
                ["doc-A"],
                ["doc-A", "doc-X"],
                ["doc-A", "doc-X"],
                0.0,
            ),
            primary_k=1,
        )

        self.assertEqual(result, ["reranker_unchanged"])

    def test_rejects_primary_k_missing_from_metric_deltas(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "primary_k must be included in k_values",
        ):
            generate_failure_tags(
                query_result(["doc-A"], ["doc-A"], ["doc-A"], 0.0),
                primary_k=5,
            )


def summarized_query_result(
    query_id: str,
    failure_tags: list[str],
    before_ranked_document_ids: list[str],
    after_ranked_document_ids: list[str],
    before_ndcg: float,
    after_ndcg: float,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_text": f"Question for {query_id}",
        "relevant_document_ids": ["doc-A"],
        "failure_tags": failure_tags,
        "before_reranking": {
            "ranked_document_ids": before_ranked_document_ids,
            "metrics": {"ndcg_at_k": {"1": before_ndcg}},
        },
        "after_reranking": {
            "ranked_document_ids": after_ranked_document_ids,
            "metrics": {"ndcg_at_k": {"1": after_ndcg}},
        },
        "metric_deltas": {
            "ndcg_at_k": {"1": after_ndcg - before_ndcg},
        },
    }


class SummarizeFailureTagsTests(unittest.TestCase):
    def test_summarizes_all_tags_and_builds_failure_case_list(self) -> None:
        query_results = [
            summarized_query_result(
                "query-001",
                [
                    "no_relevant_candidate",
                    "top_k_total_miss",
                    "persistent_top_k_miss",
                    "reranker_unchanged",
                ],
                [],
                [],
                0.0,
                0.0,
            ),
            summarized_query_result(
                "query-002",
                ["reranker_rescue", "reranker_improvement"],
                ["doc-X", "doc-A"],
                ["doc-A", "doc-X"],
                0.0,
                1.0,
            ),
            summarized_query_result(
                "query-003",
                ["top_k_total_miss", "reranker_regression"],
                ["doc-A", "doc-X"],
                ["doc-X", "doc-A"],
                1.0,
                0.0,
            ),
        ]

        result = summarize_failure_tags(query_results, primary_k=1)

        self.assertEqual(result["evaluated_query_count"], 3)
        self.assertEqual(result["failure_case_count"], 2)
        self.assertAlmostEqual(result["failure_case_rate"], 2 / 3)
        self.assertEqual(
            result["tag_counts"],
            {
                "no_relevant_candidate": 1,
                "partial_relevant_candidate_coverage": 0,
                "top_k_total_miss": 2,
                "top_k_incomplete_coverage": 0,
                "persistent_top_k_miss": 1,
                "reranker_regression": 1,
                "reranker_rescue": 1,
                "reranker_improvement": 1,
                "reranker_unchanged": 1,
            },
        )
        self.assertEqual(result["tag_rates"]["top_k_total_miss"], 2 / 3)
        self.assertEqual(
            [case["query_id"] for case in result["failure_cases"]],
            ["query-001", "query-003"],
        )
        self.assertEqual(
            result["failure_cases"][1]["before_top_k_document_ids"],
            ["doc-A"],
        )
        self.assertEqual(
            result["failure_cases"][1]["after_top_k_document_ids"],
            ["doc-X"],
        )
        self.assertEqual(
            result["failure_cases"][1]["ndcg_delta_at_primary_k"],
            -1.0,
        )

    def test_rejects_empty_query_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_results must not be empty"):
            summarize_failure_tags([], primary_k=1)

    def test_rejects_unknown_failure_tag(self) -> None:
        result = summarized_query_result(
            "query-001",
            ["unexpected_tag"],
            ["doc-A"],
            ["doc-A"],
            1.0,
            1.0,
        )

        with self.assertRaisesRegex(ValueError, "unknown failure tag"):
            summarize_failure_tags([result], primary_k=1)


if __name__ == "__main__":
    unittest.main()
