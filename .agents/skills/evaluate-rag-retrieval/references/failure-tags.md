# Failure Tags

Use failure tags as deterministic diagnostic observations at `primary_k`.
Allow multiple tags on one query. Do not treat any tag by itself as proof of a
root cause.

## Contents

- [Diagnostic layers](#diagnostic-layers)
- [Candidate coverage tags](#candidate-coverage-tags)
- [Top-K outcome tags](#top-k-outcome-tags)
- [Reranker direction tags](#reranker-direction-tags)
- [Failure-case inclusion](#failure-case-inclusion)
- [Common tag combinations](#common-tag-combinations)
- [Investigation routing](#investigation-routing)
- [Dataset-level statistics](#dataset-level-statistics)
- [Interpretation limits](#interpretation-limits)

## Diagnostic Layers

Read the tags in three layers:

```text
Candidate pool
  -> Did initial retrieval provide any or all relevant documents?

Top K after reranking
  -> Did relevant candidates reach the product-visible cutoff?

Reranker direction
  -> Did ordering quality improve, regress, or remain numerically unchanged?
```

The reranker receives the same candidate chunks as the initial ranking. It can
change order and optional scores but cannot add a missing relevant document.

All comparisons below use document IDs after chunk-to-document conversion and
document deduplication.

## Candidate Coverage Tags

### `no_relevant_candidate`

Condition:

```text
candidate relevant document count = 0
```

Observed fact: none of the ground-truth relevant documents exists anywhere in
the candidate pool.

Consequence: reranking cannot produce a hit because it cannot retrieve a new
candidate.

Investigate:

- Whether the relevant document exists in the indexed corpus.
- Document-ID alignment between ground truth and retrieval output.
- Metadata filters, access filters, or time filters.
- Chunking and document ingestion coverage.
- Embedding, lexical retrieval, query rewriting, and candidate-pool size.

Do not conclude that the reranker caused this failure.

### `partial_relevant_candidate_coverage`

Condition:

```text
0 < candidate relevant document count < ground-truth relevant document count
```

Observed fact: initial retrieval supplied some but not all relevant documents.

Investigate which labeled documents are absent and whether the question truly
requires all of them. Check corpus coverage, filters, candidate-pool size, and
retriever recall.

## Top-K Outcome Tags

### `top_k_total_miss`

Condition:

```text
relevant document count in after-reranking Top K = 0
```

Observed fact: the final Top K contains no ground-truth relevant document.

Interpret jointly with candidate coverage:

- With `no_relevant_candidate`, route first to retrieval or indexing.
- Without `no_relevant_candidate`, a relevant candidate exists below Top K;
  inspect ranking quality and the selected cutoff.

### `top_k_incomplete_coverage`

Condition:

```text
0 < relevant document count in after-reranking Top K
  < ground-truth relevant document count
```

Observed fact: the final Top K contains at least one but not all labeled
relevant documents.

Interpret jointly with `partial_relevant_candidate_coverage` to distinguish
missing candidates from candidates ranked below K.

When the number of ground-truth relevant documents is greater than K, complete
coverage is mathematically impossible at that cutoff. State this capacity limit
instead of automatically blaming a model.

### `persistent_top_k_miss`

Condition:

```text
before-reranking Top K has no relevant document
and
after-reranking Top K has no relevant document
```

Observed fact: reranking did not turn the original Top-K miss into a hit.

If a relevant candidate exists below K, investigate both baseline and reranker
ordering. If no relevant candidate exists, the reranker had no opportunity to
rescue the query.

## Reranker Direction Tags

Calculate direction from the query's NDCG delta at `primary_k`:

```text
NDCG delta = after-reranking NDCG - before-reranking NDCG
```

Use absolute tolerance `1e-12` around zero.

### `reranker_regression`

Condition:

```text
NDCG delta < -1e-12
```

Observed fact: the reranker reduced ranking quality at the primary cutoff.

Inspect which relevant documents moved down and which irrelevant documents
moved up. Possible hypotheses include domain mismatch, poor reranker features,
training-data gaps, or ambiguous query-document wording. Label these as
hypotheses until additional evidence is collected.

### `reranker_rescue`

Condition:

```text
before-reranking Top K has no relevant document
and
after-reranking Top K has at least one relevant document
```

Observed fact: the reranker moved an already-existing relevant candidate from
below K into Top K.

This tag never means that the reranker retrieved a new document. Treat a rescue
as a success case and consider preserving it in future regression tests.

### `reranker_improvement`

Condition:

```text
NDCG delta > 1e-12
```

Observed fact: the reranker improved binary-relevance ordering quality at the
primary cutoff. The improvement may come from moving one relevant document
earlier, moving more relevant documents into Top K, or both.

Study which query categories improve, but do not assume the same benefit
generalizes to untested categories.

### `reranker_unchanged`

Condition:

```text
absolute NDCG delta <= 1e-12
```

Observed fact: NDCG at the primary cutoff did not change numerically.

This tag is neither inherently good nor bad:

- Alone on a successful Top-K ranking, it describes stable quality.
- With `persistent_top_k_miss`, it describes an unresolved failure.
- At the dataset level, unchanged queries may coexist with improvements and
  regressions on other queries.

## Failure-case Inclusion

Include a query in `failure_cases` when it has at least one of these tags:

```text
no_relevant_candidate
partial_relevant_candidate_coverage
top_k_total_miss
top_k_incomplete_coverage
persistent_top_k_miss
reranker_regression
```

Do not include a query solely because it has:

```text
reranker_rescue
reranker_improvement
reranker_unchanged
```

Keep all tags on an included failure case so the report retains context. For
each case, use the supplied query, relevant document IDs, before and after Top K
document IDs, primary K, NDCG values, and NDCG delta as evidence.

## Common Tag Combinations

### Candidate pool cannot support an answer

```text
no_relevant_candidate
top_k_total_miss
persistent_top_k_miss
reranker_unchanged
```

Meaning: initial retrieval supplied no labeled relevant document. Both NDCG
values remain zero, so the reranker is numerically unchanged but had no relevant
candidate to promote.

Route first to corpus, indexing, filtering, chunking, or retrieval checks.

### Relevant candidate remains below K

```text
top_k_total_miss
persistent_top_k_miss
reranker_unchanged
```

Meaning: at least one relevant document exists in the full candidate pool, but
neither ranking places one inside Top K.

Route to baseline and reranker ordering analysis, then examine whether K is
appropriate for the product.

### Reranker pushes a relevant document out of Top K

```text
top_k_total_miss
reranker_regression
```

Meaning: the before-reranking Top K had relevance, but the after-reranking Top K
has none and NDCG decreased. Prioritize this case because reranking harmed an
originally successful result.

### Reranker rescues an existing candidate

```text
reranker_rescue
reranker_improvement
```

Meaning: a relevant candidate moved from below K into Top K and NDCG increased.
Treat this as a success pattern, not a failure case.

### Partial multi-document recovery

```text
partial_relevant_candidate_coverage
top_k_incomplete_coverage
reranker_rescue
reranker_improvement
```

Meaning: the reranker improved ordering and produced a Top-K hit, but initial
retrieval never supplied every labeled relevant document. Separate the positive
reranker effect from the unresolved candidate-recall limitation.

## Investigation Routing

Use tags to select a first investigation path, not a final diagnosis:

| Evidence pattern | First investigation area |
|---|---|
| No or partial relevant candidates | Corpus, indexing, filters, chunking, retriever, candidate depth |
| Relevant candidates exist but miss Top K | Baseline ranking, reranker, cutoff K |
| Reranker regression | Reranker model, features, prompt, training data, domain fit |
| Rescue or improvement | Success-pattern analysis and regression-test preservation |
| Incomplete coverage with more relevant docs than K | Evaluation cutoff and product capacity |

Prioritize by both severity and prevalence. A rare regression on a safety-
critical query may matter more than a common low-impact issue.

For every recommendation, state:

1. The observed tag and supporting ranking evidence.
2. The possible cause as a hypothesis.
3. A concrete check or controlled experiment.
4. The expected evidence that would confirm or reject the hypothesis.

## Dataset-level Statistics

`tag_counts` includes every known tag in fixed order, including tags with zero
occurrences. `tag_rates` divides each count by `evaluated_query_count`.

Because tags are multi-label, the sum of tag counts or rates may exceed the
number of queries or 100 percent. Do not interpret tags as mutually exclusive
categories.

`failure_case_rate` is:

```text
failure_case_count / evaluated_query_count
```

One query counts once as a failure case even when it has several failure tags.

## Interpretation Limits

- All tags depend on the supplied ground truth. Incomplete or incorrect labels
  can produce misleading diagnoses.
- Candidate coverage is measured against labeled document IDs, not semantic
  judgment made by the evaluator.
- Tags use `primary_k`; a query may improve at one K and regress at another.
- `top_k_incomplete_coverage` can be unavoidable when relevant-document count
  exceeds K.
- NDCG direction uses binary relevance and does not represent graded relevance.
- A score warning does not affect tags because ranking array order is
  authoritative.
- Tags evaluate retrieval and reranking behavior, not whether a generated
  answer is correct, faithful, safe, or complete.
