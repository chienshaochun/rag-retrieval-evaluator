import unittest

from rag_retrieval_evaluator.pairing import pair_query_records


def ground_truth_query(query_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_text": f"Question for {query_id}",
        "relevant_document_ids": [f"doc-for-{query_id}"],
    }


def retrieval_result(query_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "candidates": [],
        "before_reranking": [],
        "after_reranking": [],
    }


class PairQueryRecordsTests(unittest.TestCase):
    def test_pairs_all_queries_by_query_id(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001"), ground_truth_query("query-002")],
        }
        retrieval_results = {
            "schema_version": "1.0",
            "results": [retrieval_result("query-002"), retrieval_result("query-001")],
        }

        result = pair_query_records(ground_truth, retrieval_results)

        self.assertEqual(
            [item["query_id"] for item in result["matched_queries"]],
            ["query-001", "query-002"],
        )
        self.assertEqual(result["missing_retrieval_result_query_ids"], [])
        self.assertEqual(result["missing_ground_truth_query_ids"], [])
        self.assertEqual(
            result["coverage"],
            {
                "ground_truth_query_count": 2,
                "retrieval_result_query_count": 2,
                "matched_query_count": 2,
                "result_coverage": 1.0,
                "ground_truth_coverage": 1.0,
            },
        )

    def test_reports_queries_missing_from_each_side(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001"), ground_truth_query("query-002")],
        }
        retrieval_results = {
            "schema_version": "1.0",
            "results": [retrieval_result("query-001"), retrieval_result("query-003")],
        }

        result = pair_query_records(ground_truth, retrieval_results)

        self.assertEqual(
            [item["query_id"] for item in result["matched_queries"]],
            ["query-001"],
        )
        self.assertEqual(
            result["missing_retrieval_result_query_ids"],
            ["query-002"],
        )
        self.assertEqual(
            result["missing_ground_truth_query_ids"],
            ["query-003"],
        )
        self.assertEqual(result["coverage"]["result_coverage"], 0.5)
        self.assertEqual(result["coverage"]["ground_truth_coverage"], 0.5)

    def test_explicit_empty_retrieval_result_is_still_matched(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001")],
        }
        empty_result = retrieval_result("query-001")
        retrieval_results = {
            "schema_version": "1.0",
            "results": [empty_result],
        }

        result = pair_query_records(ground_truth, retrieval_results)

        self.assertEqual(len(result["matched_queries"]), 1)
        self.assertIs(
            result["matched_queries"][0]["retrieval_result"],
            empty_result,
        )
        self.assertEqual(result["coverage"]["result_coverage"], 1.0)

    def test_empty_results_produce_zero_result_coverage(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("query-001"), ground_truth_query("query-002")],
        }
        retrieval_results = {"schema_version": "1.0", "results": []}

        result = pair_query_records(ground_truth, retrieval_results)

        self.assertEqual(result["matched_queries"], [])
        self.assertEqual(
            result["missing_retrieval_result_query_ids"],
            ["query-001", "query-002"],
        )
        self.assertEqual(result["coverage"]["result_coverage"], 0.0)
        self.assertIsNone(result["coverage"]["ground_truth_coverage"])

    def test_query_id_matching_is_case_sensitive(self) -> None:
        ground_truth = {
            "schema_version": "1.0",
            "queries": [ground_truth_query("Query-001")],
        }
        retrieval_results = {
            "schema_version": "1.0",
            "results": [retrieval_result("query-001")],
        }

        result = pair_query_records(ground_truth, retrieval_results)

        self.assertEqual(result["matched_queries"], [])
        self.assertEqual(
            result["missing_retrieval_result_query_ids"],
            ["Query-001"],
        )
        self.assertEqual(
            result["missing_ground_truth_query_ids"],
            ["query-001"],
        )


if __name__ == "__main__":
    unittest.main()
