# Metric Definitions

Interpret all metrics at the document level with binary relevance. Convert the
ranked chunks to document IDs first and keep only the first occurrence of each
document. A higher value is better for every quality metric defined here.

## Contents

- [Notation and evaluation unit](#notation-and-evaluation-unit)
- [Hit at K](#hit-at-k)
- [Precision at K](#precision-at-k)
- [Recall at K](#recall-at-k)
- [Reciprocal rank and MRR](#reciprocal-rank-and-mrr)
- [NDCG at K](#ndcg-at-k)
- [Depth coverage at K](#depth-coverage-at-k)
- [Dataset-level macro metrics](#dataset-level-macro-metrics)
- [Before-after deltas](#before-after-deltas)
- [Worked example](#worked-example)
- [Interpretation limits](#interpretation-limits)

## Notation and Evaluation Unit

Use the following notation for one query:

- `K`: the number of leading ranking positions to evaluate.
- `TopK`: the first K unique document IDs in the ranking, or all available
  documents when the ranking is shorter than K.
- `R`: the set of relevant document IDs in ground truth.
- `relevant_in_top_k`: the number of document IDs in both `TopK` and `R`.
- `rank_first_relevant`: the one-based position of the first relevant document
  in the full deduplicated document ranking.

Require each K to be a unique positive integer. Require at least one relevant
ground-truth document for every evaluated query.

When several chunks come from one document, count that document only once at
the position of its highest-ranked chunk. This prevents repeated chunks from
the same source from inflating document-level retrieval quality.

## Hit at K

Formula:

```text
Hit@K = 1  if relevant_in_top_k >= 1
Hit@K = 0  otherwise
```

Physical meaning: answer whether the retriever placed at least one usable
document inside the first K document positions.

Hit@K does not measure how many relevant documents were found or where inside
Top K they appeared. One relevant document at rank 1 and one at rank K both
produce a hit of 1.

## Precision at K

Formula:

```text
Precision@K = relevant_in_top_k / K
```

The denominator is always K, even when fewer than K unique documents are
available. For example, returning two relevant documents when K is 5 produces
`Precision@5 = 2 / 5 = 0.4`, not 1.0.

Physical meaning: estimate how much of the fixed Top-K result capacity is
occupied by relevant documents. A low value means many positions delivered to
the downstream consumer are irrelevant or unfilled.

## Recall at K

Formula:

```text
Recall@K = relevant_in_top_k / number_of_ground_truth_relevant_documents
```

The denominator comes from `relevant_document_ids`, not K. When ground truth
contains four relevant documents and Top K contains one of them, recall is
`1 / 4 = 0.25`.

Physical meaning: measure how much of the known relevant evidence was recovered
within Top K. This is especially important for questions that require evidence
from multiple documents.

## Reciprocal Rank and MRR

Single-query formula:

```text
ReciprocalRank = 1 / rank_first_relevant
```

Return 0 when no relevant document appears anywhere in the available ranking.
This metric is not limited by K in the current evaluator.

Examples:

| First relevant position | Reciprocal rank |
|---:|---:|
| 1 | 1.0 |
| 2 | 0.5 |
| 4 | 0.25 |
| Not present | 0.0 |

Dataset-level formula:

```text
MRR = mean of every evaluated query's ReciprocalRank
```

Physical meaning: measure how quickly a user or downstream generator reaches
the first relevant document. MRR ignores additional relevant documents after
the first one.

## NDCG at K

Use binary relevance: a relevant document has gain 1 and an irrelevant document
has gain 0.

Discounted gain at one-based rank `i`:

```text
discount(i) = 1 / log2(i + 1)
```

Actual discounted cumulative gain:

```text
DCG@K = sum of discount(i) for relevant documents found at ranks 1 through K
```

Ideal discounted cumulative gain:

```text
ideal_count = min(K, number_of_ground_truth_relevant_documents)
IDCG@K = sum of discount(i) for i from 1 through ideal_count
```

Normalized score:

```text
NDCG@K = DCG@K / IDCG@K
```

Physical meaning: reward relevant documents for appearing early while still
crediting multiple relevant documents. NDCG@K equals 1 when all relevance that
can ideally fit within K occupies the earliest possible positions.

Because relevance is binary in version 1, NDCG distinguishes position but does
not distinguish degrees such as "highly relevant" and "partly relevant."

## Depth Coverage at K

Formula:

```text
DepthCoverage@K = min(number_of_available_unique_documents, K) / K
```

Physical meaning: show whether enough unique documents exist to fill the K
positions being evaluated. If only two unique documents are available at K=5,
depth coverage is `2 / 5 = 0.4`.

Depth coverage is not a relevance metric. A value of 1 means the result is deep
enough, not that it is correct. A shallow ranking can lower Precision@K because
Precision still uses K as its denominator.

The before- and after-reranking lists share one candidate pool, so the evaluator
records depth coverage once rather than treating it as a reranker quality
change.

## Dataset-level Macro Metrics

Calculate each query first, then take an unweighted arithmetic mean across all
matched queries:

| Output field | Definition |
|---|---|
| `hit_rate_at_k` | Mean of single-query `Hit@K`. |
| `macro_precision_at_k` | Mean of single-query `Precision@K`. |
| `macro_recall_at_k` | Mean of single-query `Recall@K`. |
| `mrr` | Mean of single-query reciprocal ranks. |
| `macro_ndcg_at_k` | Mean of single-query `NDCG@K`. |
| `macro_depth_coverage_at_k` | Mean of single-query depth coverage. |

Every matched query receives equal weight, regardless of its number of relevant
documents. Explicit empty retrieval results remain matched and contribute zero
quality metrics. Do not exclude zero-result queries from the average.

Queries missing from either file do not enter metric averages; report their
absence through query coverage instead.

## Before-after Deltas

Calculate every query-level and aggregate delta as:

```text
delta = after_reranking - before_reranking
```

Interpret the sign as:

- Positive: the measured quality improved.
- Negative: the measured quality regressed.
- Zero: the aggregate or query metric did not change numerically.

A zero aggregate delta does not prove that every query was unchanged. Positive
and negative query deltas may cancel. Inspect query-level tags and failure cases
before concluding that a reranker had no effect.

`primary_k` selects the cutoff used for failure classification. It does not
change formulas for the other requested K values.

## Worked Example

Given:

```text
ranked documents = [doc-X, doc-A, doc-B]
relevant documents = {doc-A, doc-B}
K = 2
```

The first two documents contain one relevant document:

```text
Hit@2       = 1
Precision@2 = 1 / 2 = 0.5
Recall@2    = 1 / 2 = 0.5
RR          = 1 / 2 = 0.5
```

For NDCG:

```text
DCG@2  = 1 / log2(2 + 1) = 0.6309297536
IDCG@2 = 1 / log2(1 + 1) + 1 / log2(2 + 1)
       = 1.6309297536
NDCG@2 = 0.6309297536 / 1.6309297536
       = 0.3868528072
```

Three unique documents are available, so `DepthCoverage@2 = 1.0`.

At K=5, both relevant documents are recovered but only three positions can be
filled:

```text
Precision@5     = 2 / 5 = 0.4
Recall@5        = 2 / 2 = 1.0
DepthCoverage@5 = 3 / 5 = 0.6
```

## Interpretation Limits

- These metrics evaluate retrieval ranking, not generated-answer correctness,
  faithfulness, hallucination, latency, or cost.
- Metric quality depends on ground-truth completeness. Missing relevant labels
  can make useful retrieved documents look irrelevant.
- Compare before and after values only at the same K. Changing K changes the
  product question being measured.
- High Recall with low Precision may mean relevant evidence is present among
  many irrelevant results.
- High Hit Rate or MRR can hide incomplete multi-document coverage; inspect
  Recall and NDCG as well.
- High NDCG does not prove that the underlying documents contain a correct or
  current answer; it only reflects agreement with the supplied relevance labels.
