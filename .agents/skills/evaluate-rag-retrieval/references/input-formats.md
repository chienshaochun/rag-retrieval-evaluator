# Input Formats

Use schema version `1.0` for both input files. Keep the ground-truth and
retrieval-result files separate, and match records by exact, case-sensitive
`query_id` values.

## Contents

- [Ground-truth file](#ground-truth-file)
- [Retrieval-results file](#retrieval-results-file)
- [Cross-file and reranking rules](#cross-file-and-reranking-rules)
- [Empty retrieval results](#empty-retrieval-results)
- [Validation behavior](#validation-behavior)

## Ground-truth File

The root must be a JSON object.

| Field | Required | Type | Rules |
|---|---|---|---|
| `schema_version` | Yes | string | Must equal `"1.0"`. |
| `dataset_name` | No | string | Must be non-blank when present. |
| `queries` | Yes | array | Must contain at least one query object. |

Each item in `queries` has this shape:

| Field | Required | Type | Rules |
|---|---|---|---|
| `query_id` | Yes | string | Must be non-blank and unique within the file. |
| `query_text` | Yes | string | Must be non-blank. |
| `relevant_document_ids` | Yes | array of strings | Must contain at least one unique, non-blank document ID. |
| `metadata` | No | object | May contain dataset-specific labels or grouping data. |

`relevant_document_ids` uses binary relevance. Every listed document is
relevant; the format does not assign graded relevance values. Multiple relevant
documents are allowed for one query.

Example:

```json
{
  "schema_version": "1.0",
  "dataset_name": "industrial-maintenance-qa",
  "queries": [
    {
      "query_id": "query-001",
      "query_text": "How do I reset the pump alarm?",
      "relevant_document_ids": ["doc-A", "doc-B"],
      "metadata": {
        "category": "pump",
        "language": "en"
      }
    },
    {
      "query_id": "query-002",
      "query_text": "What causes motor overheating?",
      "relevant_document_ids": ["doc-C"]
    }
  ]
}
```

## Retrieval-results File

The root must be a JSON object.

| Field | Required | Type | Rules |
|---|---|---|---|
| `schema_version` | Yes | string | Must equal `"1.0"`. |
| `run_name` | No | string | Must be non-blank when present. |
| `results` | Yes | array | May be empty when no query results are available. |

Each item in `results` has this shape:

| Field | Required | Type | Rules |
|---|---|---|---|
| `query_id` | Yes | string | Must be non-blank and unique within the file. |
| `candidates` | Yes | array | Contains the complete candidate chunk pool. |
| `before_reranking` | Yes | array | Orders every candidate chunk before reranking. |
| `after_reranking` | Yes | array | Orders the same candidate chunks after reranking. |

Each item in `candidates` has this shape:

| Field | Required | Type | Rules |
|---|---|---|---|
| `chunk_id` | Yes | string | Must be non-blank and unique within this candidate pool. |
| `document_id` | Yes | string | Must be non-blank. Multiple chunks may belong to one document. |
| `chunk_text` | No | string | May preserve the retrieved text for later inspection. |
| `metadata` | No | object | May contain source, page, section, or other chunk data. |

Each ranking item has this shape:

| Field | Required | Type | Rules |
|---|---|---|---|
| `chunk_id` | Yes | string | Must be non-blank and appear once in that ranking. |
| `score` | No | finite number | Invalid values produce a warning rather than a validation error. |

The array order is authoritative. Metrics use the listed order, not `score`.
Scores from the retriever and reranker do not need to be on the same numeric
scale.

Example:

```json
{
  "schema_version": "1.0",
  "run_name": "bm25-vs-cross-encoder-v1",
  "results": [
    {
      "query_id": "query-001",
      "candidates": [
        {
          "chunk_id": "chunk-X1",
          "document_id": "doc-X",
          "chunk_text": "General alarm overview"
        },
        {
          "chunk_id": "chunk-A1",
          "document_id": "doc-A",
          "chunk_text": "Pump alarm reset procedure",
          "metadata": {"page": 12}
        },
        {
          "chunk_id": "chunk-B1",
          "document_id": "doc-B",
          "chunk_text": "Pump restart safety checks"
        }
      ],
      "before_reranking": [
        {"chunk_id": "chunk-X1", "score": 8.7},
        {"chunk_id": "chunk-A1", "score": 7.9},
        {"chunk_id": "chunk-B1", "score": 7.1}
      ],
      "after_reranking": [
        {"chunk_id": "chunk-A1", "score": 0.96},
        {"chunk_id": "chunk-B1", "score": 0.82},
        {"chunk_id": "chunk-X1", "score": 0.15}
      ]
    }
  ]
}
```

## Cross-file and Reranking Rules

- Match ground-truth queries and retrieval results by exact `query_id`.
  `query-001` and `Query-001` are different IDs.
- Permit missing results and extra result records. Report both directions as
  query coverage instead of silently discarding the mismatch.
- Interpret each ground-truth ID as a document ID, not a chunk ID.
- Permit multiple candidate chunks to map to the same `document_id`. Convert
  chunk rankings to document rankings and keep only the first occurrence of
  each document before calculating metrics.
- Require `candidates`, `before_reranking`, and `after_reranking` to contain the
  exact same set of `chunk_id` values for each query.
- Allow only ranking order and optional scores to change between before and
  after. A reranker does not retrieve new candidates or remove candidates.

## Empty Retrieval Results

Represent a query that was executed but returned no candidates explicitly:

```json
{
  "query_id": "query-002",
  "candidates": [],
  "before_reranking": [],
  "after_reranking": []
}
```

This record still matches its ground-truth query and contributes zero retrieval
metrics. It is different from omitting `query-002`, which lowers query coverage.

Represent a run containing no result records as:

```json
{
  "schema_version": "1.0",
  "results": []
}
```

If no result `query_id` matches any ground-truth query, the pipeline returns
`status: "no_matched_queries"` and does not invent aggregate metrics.

## Validation Behavior

Reject the dataset when required structure is invalid, IDs are blank or
duplicated, or ranking candidate sets differ. Collect multiple structural
errors in one validation result when possible.

Keep evaluation running when only a ranking `score` is missing or invalid.
Record a warning and continue to use array order as the ranking source.

Common invalid conditions include:

- An unsupported or missing `schema_version`.
- An empty ground-truth `queries` array.
- Duplicate `query_id`, `relevant_document_ids`, candidate `chunk_id`, or
  ranking `chunk_id` values within their required scopes.
- Blank IDs or query text.
- Non-object metadata values.
- A before- or after-reranking list that adds, removes, or replaces a candidate
  chunk.
