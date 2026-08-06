import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPT = (
    PROJECT_ROOT
    / ".agents"
    / "skills"
    / "evaluate-rag-retrieval"
    / "scripts"
    / "evaluate_rag.py"
)


class SkillEntrypointTests(unittest.TestCase):
    def test_runs_from_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            copied_scripts = directory / "skill-scripts"
            shutil.copytree(
                SKILL_SCRIPT.parent,
                copied_scripts,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            copied_skill_script = copied_scripts / "evaluate_rag.py"
            ground_truth_path = directory / "ground_truth.json"
            retrieval_results_path = directory / "retrieval_results.json"
            output_path = directory / "evaluation.json"

            ground_truth_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "queries": [
                            {
                                "query_id": "query-001",
                                "query_text": "How do I reset the alarm?",
                                "relevant_document_ids": ["doc-A"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            retrieval_results_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
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
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(copied_skill_script),
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
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.is_file())
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["configuration"]["k_values"], [1, 5])
            self.assertEqual(result["evaluation"]["evaluated_query_count"], 1)


if __name__ == "__main__":
    unittest.main()
