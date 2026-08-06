# RAG Retrieval Evaluator

一個可由 Codex Skill 驅動的 RAG 檢索評估工具。它讀取 ground truth 與
retrieval results，驗證輸入格式、計算檢索指標、比較 reranker 前後差異、
標記失敗案例，最後產生結構化 JSON 與繁體中文 Markdown 報告。

## 專案目標

這個專案專注於回答以下問題：

- Retriever 是否把相關文件放進候選集合？
- 相關文件是否出現在指定的 Top K？
- Reranker 讓排序改善、退步，還是只在平均值上看似沒有變化？
- 哪些失敗應該先檢查 retrieval/index，哪些應該檢查 reranker？
- 如何把客觀指標轉換成有證據的工程改善建議？

它不負責執行線上檢索、訓練 reranker，或評估生成答案的正確性、忠實度與
hallucination。

## 核心設計

```text
Ground truth JSON + Retrieval results JSON
                    ↓
          Python deterministic evaluator
                    ↓
     Validation、metrics、comparison、failure tags
                    ↓
               evaluation.json
                    ↓
       Codex Agent 依 references 解讀
                    ↓
            evaluation.report.md
```

Python 負責可重現的量測；Agent 負責自然語言分析。報告中的數值以
`evaluation.json` 為唯一來源，不由 Agent 重新計算。

## 功能

- 驗證 ground truth 與 retrieval-result JSON。
- 依 exact、case-sensitive `query_id` 配對資料。
- 支援一個 Query 對應多個相關文件。
- 使用 binary relevance 與 document-level 評估。
- 將重複 chunk 轉換並去重成 document ranking。
- 計算 Hit@K、Precision@K、Recall@K、MRR 與 NDCG@K。
- 比較 reranker 前後的 Query-level 與 aggregate 指標。
- 偵測 reranker rescue、improvement、regression 與 persistent miss。
- 分開呈現 Query 配對覆蓋率與 ranking depth coverage。
- 保存結構化 JSON 與人類可讀 Markdown 報告。

## 環境需求

- Python 3.10 以上
- 不需要第三方 Python 套件
- 使用完整 Skill 工作流與 Markdown 報告時，需要 Codex

## 使用 Codex Skill

以這個 Repository 作為 Codex 的目前專案，然後明確呼叫
`$evaluate-rag-retrieval`。Skill 位於
[`.agents/skills/evaluate-rag-retrieval`](.agents/skills/evaluate-rag-retrieval)。

可使用內建 E2E 範例測試：

```text
請使用 $evaluate-rag-retrieval 評估以下 RAG 檢索資料：

Ground truth：
examples/e2e/ground_truth.json

Retrieval results：
examples/e2e/retrieval_results.json

請使用 k_values 1、3、5，primary_k 3。

請將結構化結果儲存至：
evaluation_runs/demo-01/evaluation.json

若輸出目錄不存在請建立。請依 Skill 規格產生繁體中文報告，並保存
Markdown 檔案。不要自行重新計算 evaluation JSON 中的數值。
```

未指定報告路徑時，Skill 會把報告放在 JSON 旁邊：

```text
evaluation_runs/demo-01/evaluation.json
evaluation_runs/demo-01/evaluation.report.md
```

`evaluation_runs/` 預設由 Git 忽略，適合保存日常本機實驗。值得公開或長期
保留的結果可以整理到 `examples/`。

## 直接執行 Python Evaluator

先建立輸出目錄，再從 Repository 根目錄執行：

```text
python .agents/skills/evaluate-rag-retrieval/scripts/evaluate_rag.py --ground-truth examples/e2e/ground_truth.json --retrieval-results examples/e2e/retrieval_results.json --output evaluation_runs/demo-01/evaluation.json --k-values 1 3 5 --primary-k 3
```

CLI 只產生 deterministic `evaluation.json`。Markdown 報告由 Codex Skill
讀取 JSON 後產生，不由 Python 自然語言生成。

## 輸入資料

### Ground truth

```json
{
  "schema_version": "1.0",
  "dataset_name": "maintenance-qa",
  "queries": [
    {
      "query_id": "query-001",
      "query_text": "How do I reset the pump alarm?",
      "relevant_document_ids": ["doc-A", "doc-B"]
    }
  ]
}
```

### Retrieval results

```json
{
  "schema_version": "1.0",
  "run_name": "hybrid-plus-reranker-v1",
  "results": [
    {
      "query_id": "query-001",
      "candidates": [
        {"chunk_id": "chunk-A1", "document_id": "doc-A"}
      ],
      "before_reranking": [{"chunk_id": "chunk-A1"}],
      "after_reranking": [{"chunk_id": "chunk-A1"}]
    }
  ]
}
```

`before_reranking` 與 `after_reranking` 必須包含完全相同的 candidate
chunk IDs，只允許順序與選填 score 改變。完整規格請參考
[input-formats.md](.agents/skills/evaluate-rag-retrieval/references/input-formats.md)。

## 指標

| 指標 | 物理意義 |
|---|---|
| Hit@K | Top K 是否至少出現一份相關文件 |
| Precision@K | Top K 中有多少比例是相關文件 |
| Recall@K | Ground truth 相關文件有多少比例進入 Top K |
| MRR | 第一份相關文件出現得多前面 |
| NDCG@K | 相關文件是否集中在較前面的排名位置 |

所有指標都是越高越好。完整公式與限制請參考
[metric-definitions.md](.agents/skills/evaluate-rag-retrieval/references/metric-definitions.md)。

## 失敗分析

失敗標籤分成三個層級：

```text
Candidate coverage
→ Retriever 是否提供相關候選文件

Top-K outcome
→ 相關候選是否進入產品實際使用的 Top K

Reranker direction
→ Reranker 是否改善、退步或數值不變
```

同一個 Query 可以同時具有多個標籤，因此 tag rates 相加可能超過 100%。
完整條件與調查方向請參考
[failure-tags.md](.agents/skills/evaluate-rag-retrieval/references/failure-tags.md)。

## E2E 範例

[`examples/e2e`](examples/e2e) 保存一組完整且可重現的範例：

- [Ground truth](examples/e2e/ground_truth.json)
- [Retrieval results](examples/e2e/retrieval_results.json)
- [Evaluation JSON](examples/e2e/evaluation.json)
- [繁體中文報告](examples/e2e/evaluation.report.md)

這組資料刻意包含 reranker rescue、reranker regression、缺少相關 candidate、
persistent Top-K miss、Query 配對缺漏與 score warning。

## 專案結構

```text
rag-retrieval-evaluator/
├── .agents/skills/evaluate-rag-retrieval/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   ├── references/
│   └── scripts/
├── examples/e2e/
├── evaluation_runs/       # 本機產生，Git ignored
├── tests/
├── .gitignore
└── README.md
```

## 執行測試

```text
python -m unittest discover -s tests
```

測試涵蓋 validation、pairing、metrics、comparison、failure analysis、pipeline、
CLI，以及把 Skill scripts 複製到 Repository 外後執行的自包含測試。

驗證 Skill 結構時可執行 `skill-creator` 內附的 `quick_validate.py`。

## 重要限制

- 結果品質依賴 ground truth 的正確性與完整性。
- 指標只涵蓋 exact `query_id` 成功配對的 Query。
- Binary relevance 不區分相關程度。
- 結論依賴選擇的 K 與 `primary_k`。
- Retrieval relevance 不代表最終答案正確、忠實或沒有 hallucination。
- 本工具不評估 latency、cost、資料新鮮度或業務成果。
