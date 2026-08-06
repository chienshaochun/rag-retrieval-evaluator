"""Command-line adapter for file-based RAG retrieval evaluation."""

import argparse
import json
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Sequence

from .pipeline import run_evaluation_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-rag",
        description="Evaluate RAG retrieval results against ground truth.",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--retrieval-results",
        required=True,
        type=Path,
        help="Path to the retrieval results JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the structured evaluation JSON will be written.",
    )
    parser.add_argument(
        "--k-values",
        required=True,
        nargs="+",
        type=int,
        metavar="K",
        help="One or more positive evaluation cutoffs, for example: 1 3 5.",
    )
    parser.add_argument(
        "--primary-k",
        required=True,
        type=int,
        help="The K cutoff used for failure classification.",
    )
    return parser


def _load_json(path: Path, label: str) -> object:
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ValueError(f"cannot read {label} file {str(path)!r}: {error}") from error

    try:
        return json.loads(raw_text)
    except JSONDecodeError as error:
        raise ValueError(
            f"{label} file {str(path)!r} is not valid JSON: "
            f"line {error.lineno}, column {error.colno}"
        ) from error


def _write_json(path: Path, result: dict[str, object]) -> None:
    serialized = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    try:
        path.write_text(serialized + "\n", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot write output file {str(path)!r}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator CLI and return a process exit code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    input_paths = (arguments.ground_truth, arguments.retrieval_results)
    resolved_output_path = arguments.output.resolve()
    if any(resolved_output_path == input_path.resolve() for input_path in input_paths):
        print("error: output path must differ from both input paths", file=sys.stderr)
        return 1

    try:
        ground_truth_data = _load_json(arguments.ground_truth, "ground truth")
        retrieval_results_data = _load_json(
            arguments.retrieval_results,
            "retrieval results",
        )
        result = run_evaluation_pipeline(
            ground_truth_data,
            retrieval_results_data,
            arguments.k_values,
            arguments.primary_k,
        )
        _write_json(arguments.output, result)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Evaluation written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
