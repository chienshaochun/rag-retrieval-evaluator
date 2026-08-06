---
name: evaluate-rag-retrieval
description: >-
  Evaluate RAG retrieval and reranking results against ground truth using
  deterministic Python metrics and failure analysis. Use when Codex needs to
  validate ground-truth or retrieval-result JSON, calculate Hit@K,
  Precision@K, Recall@K, MRR, or NDCG, compare rankings before and after
  reranking, identify failed queries or reranker regressions, or produce an
  evidence-based retrieval evaluation report and improvement recommendations.
  Do not use for building a RAG system, executing live retrieval, training a
  reranker, or evaluating generated-answer correctness, faithfulness, or
  hallucination unless retrieval-ranking evaluation is also requested.
---

# Evaluate RAG Retrieval

## Workflow

1. Locate the ground-truth JSON file and retrieval-results JSON file.
2. Confirm `k_values` and `primary_k` with the user. Ask for them when missing;
   do not silently choose evaluation cutoffs.
3. Resolve bundled paths relative to this `SKILL.md` file.
4. Run `scripts/evaluate_rag.py` with absolute input and output paths.
5. Stop on a nonzero exit code. Report the validation or file error and do not
   create an evaluation narrative from invalid data.
6. Inspect `status`, validation warnings, and query coverage before interpreting
   any metric.
7. Compare aggregate metrics before and after reranking, then inspect query tags
   and failure cases for changes hidden by averages.
8. Produce the report in the user's language. Separate observed facts, possible
   causes, recommended checks, and improvement actions.

## Run the Evaluator

Use this command shape:

```text
python <skill-dir>/scripts/evaluate_rag.py \
  --ground-truth <ground-truth.json> \
  --retrieval-results <retrieval-results.json> \
  --output <evaluation.json> \
  --k-values <K1> <K2> ... \
  --primary-k <K>
```

Keep the output path different from both input paths. Treat the generated JSON
as the sole source of metric values; do not recalculate or alter metrics by
hand.

## Interpret the Result

- If `status` is `no_matched_queries`, explain the query-ID mismatch and stop
  metric interpretation. Do not describe missing metrics as zero performance.
- Report validation warnings and incomplete query coverage before performance
  conclusions. State when conclusions apply only to the matched subset.
- Read aggregate before/after values together with `metric_deltas`. Do not infer
  that the reranker had no effect merely because an aggregate delta is zero;
  improvements and regressions may cancel.
- Use `primary_k` when explaining failure tags and Top-K outcomes.
- Treat `failure_tags` as diagnostic observations, not proven root causes.
- Remember that reranking only reorders the existing candidate set. Interpret
  `reranker_rescue` as moving an existing relevant candidate into Top K, never
  as retrieving a new document.
- Keep successful rescue and improvement cases out of the failure worklist
  unless another failure tag is also present.

## Write the Report

Read `references/report-guidance.md` before writing a user-facing report. Cover
the evaluation scope, data quality, aggregate comparison, reranker impact,
failure distribution, representative cases, recommendations, and limitations.

For every recommendation:

1. State the measured evidence.
2. Label the possible cause as a hypothesis.
3. Recommend a concrete check or experiment.
4. Avoid claiming that a tag alone proves which component is defective.

Keep the report proportional to the dataset size. Warn against generalizing
from very small datasets or low query coverage.

## Load References Selectively

- Read `references/input-formats.md` when creating, explaining, or repairing
  ground-truth and retrieval-result JSON.
- Read `references/metric-definitions.md` when explaining formulas, physical
  meaning, denominators, or metric limitations.
- Read `references/failure-tags.md` when failure cases exist or the user requests
  diagnosis and improvement recommendations.
- Read `references/report-guidance.md` whenever producing the final narrative
  report.

## Guardrails

- Preserve the input files and write results to a separate output path.
- Treat ranking array order as authoritative; scores are optional supporting
  data and do not replace array order.
- Require the before- and after-reranking lists to contain the same candidate
  chunk IDs. Reject candidate-set changes as invalid reranking input.
- Evaluate relevance at the document level after deduplicating repeated chunks
  from the same document.
- Do not use retrieval metrics to make claims about answer correctness,
  faithfulness, hallucination, latency, or cost unless separate evidence is
  provided.
