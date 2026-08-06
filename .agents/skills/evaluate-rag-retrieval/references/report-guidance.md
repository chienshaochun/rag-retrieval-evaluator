# Evaluation Report Guidance

Use this guide to turn the evaluator's structured JSON into a readable,
evidence-based report. Treat the JSON as the source of truth. Do not recalculate,
replace, or silently correct metric values while writing the report.

## Contents

- [Separate Analysis Order from Report Order](#separate-analysis-order-from-report-order)
- [Inspect the Evaluation Safely](#inspect-the-evaluation-safely)
- [Handle No Matched Queries](#handle-no-matched-queries)
- [Write the Seven Report Sections](#write-the-seven-report-sections)
- [Present Metric Comparisons](#present-metric-comparisons)
- [Describe Reranker Impact](#describe-reranker-impact)
- [Summarize Failure Distribution](#summarize-failure-distribution)
- [Select Representative Failure Cases](#select-representative-failure-cases)
- [Separate Facts, Hypotheses, Checks, and Actions](#separate-facts-hypotheses-checks-and-actions)
- [Route Recommendations to the Correct Layer](#route-recommendations-to-the-correct-layer)
- [State Limitations](#state-limitations)
- [Use Verdict Labels Carefully](#use-verdict-labels-carefully)
- [Markdown Report Template](#markdown-report-template)
- [No-Matched-Queries Template](#no-matched-queries-template)

## Separate Analysis Order from Report Order

Inspect the JSON in this order before writing any conclusion:

1. Read root-level `status`.
2. Read `validation_warnings`.
3. Read `pairing.coverage` and both missing-query-ID lists.
4. Read `configuration.k_values` and `configuration.primary_k`.
5. Read `evaluation.aggregate_metrics`, including before, after, and delta.
6. Read `evaluation.failure_analysis` counts and rates.
7. Read query-level `evaluation.query_results` and representative failure cases.
8. Synthesize the report only after completing the checks above.

Present the finished report in this order:

1. Executive summary
2. Evaluation scope and data quality
3. Overall metric comparison
4. Reranker impact
5. Failure distribution
6. Representative failure cases
7. Recommendations and limitations

Write the executive summary last even though it appears first. This prevents an
early impression from controlling the later evidence review.

## Inspect the Evaluation Safely

Use these root fields to establish the evaluation context:

- `dataset_name`: optional ground-truth dataset label.
- `run_name`: optional retrieval-run label.
- `configuration`: the evaluated K values and the primary diagnostic cutoff.
- `validation_warnings`: valid but potentially important data-quality concerns.
- `pairing`: query matching, missing IDs, and coverage.
- `evaluation`: metrics and failure analysis, or `null` when no query IDs match.

Report warnings before performance conclusions. When either coverage rate is
less than `1.0`, explicitly state that the metrics describe only the matched
query subset. Do not treat unmatched queries as retrieval misses because they
were not evaluated.

Keep these two coverage fields distinct:

- `result_coverage`: share of ground-truth queries with retrieval results.
- `ground_truth_coverage`: share of retrieval-result queries with ground truth;
  this may be `null` when the retrieval input has no queries.

Also distinguish query pairing coverage from
`evaluation.aggregate_metrics.macro_depth_coverage_at_k`. Pairing coverage
describes which queries could be evaluated. Depth coverage describes whether
ranked lists were long enough to reach each requested K.

## Handle No Matched Queries

When `status` is `no_matched_queries` and `evaluation` is `null`:

1. State that no exact query IDs matched.
2. List or summarize
   `pairing.missing_retrieval_result_query_ids` and
   `pairing.missing_ground_truth_query_ids`.
3. Explain that metrics were not calculated.
4. Recommend checking ID spelling, capitalization, dataset versions, and file
   pairing.
5. Stop. Do not produce metric tables, reranker verdicts, or zero-performance
   claims.

Missing metrics mean "not evaluated," not "performance equals zero."

## Write the Seven Report Sections

### 1. Executive Summary

Summarize the evaluated scope, the overall reranker direction, the most
important failure pattern, and the highest-priority next action. Keep it short
and qualify it when coverage or sample size is limited.

### 2. Evaluation Scope and Data Quality

Include:

- dataset and run names when present;
- matched and total query counts;
- both pairing coverage rates;
- requested K values and `primary_k`;
- validation warnings and missing-query counts;
- the fact that relevance is binary and document-level.

### 3. Overall Metric Comparison

Show before-reranking, after-reranking, and delta values for the requested
metrics. Include every evaluated K and visually identify `primary_k` in text or
in the table label.

### 4. Reranker Impact

Describe whether ordering quality improved, regressed, remained numerically
unchanged, or showed mixed effects. Use both aggregate metrics and query-level
tags. Explain any conflict between averages and individual cases.

### 5. Failure Distribution

Report failure-case count and rate, then important tag counts and rates. Explain
that tags are multi-label, so tag counts can sum above the query count and tag
rates can sum above 100%.

### 6. Representative Failure Cases

Choose cases that explain the dominant problems and the highest-risk reranker
changes. Provide enough evidence to reproduce each observation without dumping
every query into the main report.

### 7. Recommendations and Limitations

Prioritize concrete checks and experiments by system layer. End with the limits
of what this retrieval evaluation can establish.

## Present Metric Comparisons

Use a table shaped like this:

| Metric | Before reranking | After reranking | Delta |
|---|---:|---:|---:|
| Hit@K | value | value | value |
| Macro Precision@K | value | value | value |
| Macro Recall@K | value | value | value |
| MRR | value | value | value |
| Macro NDCG@K | value | value | value |

Read values from these objects:

- `aggregate_metrics.before_reranking`
- `aggregate_metrics.after_reranking`
- `aggregate_metrics.metric_deltas`

Display values consistently, normally to three or four decimal places. Treat
rounding as presentation only; preserve the original JSON and never overwrite
its values. Use the metric definitions reference when a metric needs a formula
or physical interpretation.

State that these metrics are higher-is-better. Do not choose an overall verdict
from only one metric. For example, unchanged Hit@K can coexist with improved
NDCG when the same relevant documents move closer to rank 1.

Report `macro_depth_coverage_at_k` separately from quality metrics. Low depth
coverage means some result lists ended before K, so the system had fewer than K
opportunities on those queries. It is not by itself evidence that the reranker
is poor.

## Describe Reranker Impact

Use the following evidence together:

- aggregate metric deltas;
- `reranker_improved`, `reranker_regression`, and `reranker_unchanged` tag
  counts;
- `reranker_rescue` cases;
- before/after Top-K document IDs in failure cases;
- query-level NDCG deltas at `primary_k`.

Remember that the reranker only reorders the initial candidate set. Never say it
"retrieved" or "found" a document that was absent from the initial candidates.
A rescue means an existing relevant candidate moved into Top K.

An aggregate delta of zero does not prove that no ranking changed. Query-level
improvements and regressions can cancel. Conversely, a positive average does
not mean every query improved.

Do not compare optional model scores as if they share a calibrated scale unless
the input provides separate evidence that they are comparable. Ranking array
order remains authoritative.

## Summarize Failure Distribution

Start with:

- `failure_analysis.evaluated_query_count`
- `failure_analysis.failure_case_count`
- `failure_analysis.failure_case_rate`

Then report the most useful entries from `tag_counts` and `tag_rates`. Use the
failure-tags reference for exact tag meanings and investigation routing.

Group tags conceptually when it improves readability:

- candidate coverage problems;
- Top-K outcome problems;
- reranker direction or regression problems.

Do not call every diagnostic tag a failure. For example, a successful rescue or
an improvement tag describes a positive change unless another failure condition
also places the query in the failure worklist.

## Select Representative Failure Cases

Prioritize cases in this order:

1. Reranker regressions, especially a relevant document pushed out of Top K.
2. Queries with `no_relevant_candidate`.
3. Persistent misses after reranking.
4. Other Top-K misses or incomplete-relevance cases.
5. Repeated examples of a dominant tag when they reveal a consistent pattern.

For each selected case, include:

- query ID and query text;
- ground-truth relevant document IDs;
- before and after Top-K document IDs;
- failure tags;
- before NDCG, after NDCG, and NDCG delta at `primary_k`;
- an observed fact, possible cause, recommended check, and next action.

When there are many cases, summarize totals and show a representative subset.
Offer the complete machine-readable JSON as the exhaustive record. If business
metadata such as product area, language, tenant, or document type is supplied
separately, use it to choose cases, but do not invent missing metadata.

## Separate Facts, Hypotheses, Checks, and Actions

Use this four-part discipline for every diagnosis:

1. **Observed fact:** Quote or paraphrase exact metrics, tags, rankings, or
   coverage evidence from the evaluation JSON.
2. **Possible cause:** Label the explanation as a hypothesis. Do not present it
   as proven by a tag.
3. **Recommended check:** Name a reproducible inspection or experiment.
4. **Action if confirmed:** State the component or configuration to change and
   the metric expected to respond.

Avoid this unsupported claim:

> The embedding model is broken.

Prefer this evidence chain:

> Observed: 25% of evaluated queries have `no_relevant_candidate`. Possible
> cause: the index, chunking, or first-stage retrieval may not expose relevant
> content. Check the affected documents' index presence and run targeted
> retrieval tests before changing the reranker.

## Route Recommendations to the Correct Layer

Route actions according to the evidence:

- **Retrieval/index layer:** investigate missing candidates, indexing gaps,
  chunking, query representation, metadata filters, candidate depth, and
  document-ID mapping.
- **Reranker layer:** investigate regressions, feature mismatch, training-data
  mismatch, truncation, score behavior, or ranking constraints only when the
  relevant document already existed in the candidate set.
- **Evaluation/configuration layer:** investigate low pairing coverage, invalid
  IDs, incomplete ground truth, insufficient result depth, unsuitable K values,
  or a small/nonrepresentative query sample.

Order recommendations by measured frequency, likely user impact, and ease of
verification. Recommend changing one layer at a time when possible so the next
evaluation can attribute the effect.

## State Limitations

Include the limitations that apply:

- Ground truth may be incomplete, inconsistent, or outdated.
- Metrics cover only exactly matched query IDs.
- Small or unrepresentative datasets do not justify broad production claims.
- Conclusions depend on the selected K values and `primary_k`.
- Binary relevance treats every relevant document equally.
- Document-level deduplication hides repeated chunks from the same document.
- Retrieval relevance does not prove generated-answer correctness,
  faithfulness, or absence of hallucination.
- Optional scores are not required and do not replace ranking order.
- Latency, cost, freshness, and business outcomes require separate evidence.

Do not generalize a reranker verdict beyond the evaluated dataset and run.

## Use Verdict Labels Carefully

Use these qualitative labels without inventing fixed thresholds:

- **Improved:** several relevant aggregate measures improve and no material
  counterevidence dominates.
- **Regressed:** relevant aggregate measures decline or important query-level
  regressions outweigh the gains.
- **Numerically unchanged:** the reported aggregate deltas are zero; still check
  query-level changes before saying the rankings were unchanged.
- **Mixed:** improvements and regressions coexist, metrics disagree, or averages
  hide offsetting query-level changes.
- **Inconclusive:** coverage, sample size, warnings, or missing evaluation data
  make a reliable direction unsupported.

Always attach the evidence behind the label. Avoid universal claims such as
"the reranker is better" when the evidence only supports "better on this
matched dataset at the evaluated K values."

## Markdown Report Template

```markdown
# RAG Retrieval Evaluation Report

## Executive Summary

[Scope-qualified verdict, main evidence, largest risk, highest-priority action]

## Evaluation Scope and Data Quality

- Dataset / run: ...
- Matched queries: ...
- Pairing coverage: ...
- K values / primary K: ...
- Warnings and missing IDs: ...

## Overall Metric Comparison

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| ... | ... | ... | ... |

Depth coverage: ...

## Reranker Impact

[Aggregate direction, rescues, regressions, cancellations, and scope]

## Failure Distribution

[Failure count/rate and important multi-label tag counts/rates]

## Representative Failure Cases

### Query: ...

- Observed fact: ...
- Possible cause: ...
- Recommended check: ...
- Action if confirmed: ...

## Recommendations and Limitations

1. [Evidence-linked action]
2. [Evidence-linked action]

Limitations: ...
```

Write the report in the user's language unless the user requests another
language. Keep technical field names in code formatting when precision helps.

## No-Matched-Queries Template

```markdown
# RAG Retrieval Evaluation Report

## Evaluation Status

No exact query IDs matched between ground truth and retrieval results, so no
retrieval or reranker metrics were calculated.

- Ground-truth queries without results: ...
- Retrieval-result queries without ground truth: ...
- Validation warnings: ...

## Recommended Checks

1. Compare query-ID spelling and capitalization.
2. Confirm both files belong to the same dataset and run version.
3. Repair the IDs or file pairing, then rerun the evaluator.
```

Stop after this diagnostic report. Do not add an empty comparison table or an
overall reranker verdict.
