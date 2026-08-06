import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rag_retrieval_evaluator.cli import main


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


class CliTests(unittest.TestCase):
    def test_reads_input_files_and_writes_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            ground_truth_path = directory / "ground_truth.json"
            retrieval_results_path = directory / "retrieval_results.json"
            output_path = directory / "evaluation.json"
            write_json(
                ground_truth_path,
                {
                    "schema_version": "1.0",
                    "dataset_name": "繁體中文測試",
                    "queries": [
                        {
                            "query_id": "query-001",
                            "query_text": "如何重設泵浦警報？",
                            "relevant_document_ids": ["doc-A"],
                        }
                    ],
                },
            )
            write_json(
                retrieval_results_path,
                {
                    "schema_version": "1.0",
                    "run_name": "reranker-v1",
                    "results": [
                        {
                            "query_id": "query-001",
                            "candidates": [
                                {
                                    "chunk_id": "chunk-A",
                                    "document_id": "doc-A",
                                }
                            ],
                            "before_reranking": [{"chunk_id": "chunk-A"}],
                            "after_reranking": [{"chunk_id": "chunk-A"}],
                        }
                    ],
                },
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--ground-truth",
                        str(ground_truth_path),
                        "--retrieval-results",
                        str(retrieval_results_path),
                        "--output",
                        str(output_path),
                        "--k-values",
                        "1",
                        "5",
                        "--primary-k",
                        "1",
                    ]
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["dataset_name"], "繁體中文測試")
            self.assertEqual(result["configuration"]["k_values"], [1, 5])
            self.assertEqual(result["evaluation"]["evaluated_query_count"], 1)

    def test_invalid_json_returns_failure_without_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            ground_truth_path = directory / "ground_truth.json"
            retrieval_results_path = directory / "retrieval_results.json"
            output_path = directory / "evaluation.json"
            ground_truth_path.write_text("{invalid", encoding="utf-8")
            write_json(
                retrieval_results_path,
                {"schema_version": "1.0", "results": []},
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--ground-truth",
                        str(ground_truth_path),
                        "--retrieval-results",
                        str(retrieval_results_path),
                        "--output",
                        str(output_path),
                        "--k-values",
                        "1",
                        "--primary-k",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("is not valid JSON", stderr.getvalue())
            self.assertFalse(output_path.exists())

    def test_rejects_output_path_that_would_overwrite_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            ground_truth_path = directory / "ground_truth.json"
            retrieval_results_path = directory / "retrieval_results.json"
            write_json(
                ground_truth_path,
                {
                    "schema_version": "1.0",
                    "queries": [
                        {
                            "query_id": "query-001",
                            "query_text": "Question",
                            "relevant_document_ids": ["doc-A"],
                        }
                    ],
                },
            )
            original_ground_truth = ground_truth_path.read_text(encoding="utf-8")
            write_json(
                retrieval_results_path,
                {"schema_version": "1.0", "results": []},
            )
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--ground-truth",
                        str(ground_truth_path),
                        "--retrieval-results",
                        str(retrieval_results_path),
                        "--output",
                        str(ground_truth_path),
                        "--k-values",
                        "1",
                        "--primary-k",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("output path must differ", stderr.getvalue())
            self.assertEqual(
                ground_truth_path.read_text(encoding="utf-8"),
                original_ground_truth,
            )


if __name__ == "__main__":
    unittest.main()
