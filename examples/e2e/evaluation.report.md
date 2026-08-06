# RAG 檢索評估報告

## 1. 摘要

本次評估結果為「混合」。在 5 筆成功配對的 Query 中，reranker
成功救援 1 筆，也使另 1 筆原本成功的結果退步。兩者在主要觀察點
`K=3` 的效果剛好抵消，因此整體 NDCG@3 維持 0.3839，不能解讀成
reranker 沒有改變排序。

目前最高優先事項是檢查 `query-002` 的 reranker regression，並另外處理
`query-003` 缺少相關候選文件的 first-stage retrieval 問題。本次資料量小且
Query 配對覆蓋率只有 83.33%，結論只適用於這次成功配對的測試子集。

## 2. 評估範圍與資料品質

- Dataset：`industrial-maintenance-e2e-demo`
- Run：`hybrid-search-plus-reranker-e2e-demo`
- Ground truth Query：6 筆
- Retrieval result Query：6 筆
- 成功配對：5 筆
- `result_coverage`：83.33%
- `ground_truth_coverage`：83.33%
- 評估 K：1、3、5
- 主要診斷點：`primary_k=3`
- Relevance：binary relevance、document-level
- Ground truth 有但 retrieval 缺少：`query-006`
- Retrieval 有但 ground truth 缺少：`query-999`

驗證警告：`query-001` 的 after-reranking 第一筆 `score` 不是有限數字。
這不會中止評估，因為排序陣列的順序才是指標計算依據，但應修正上游輸出
格式，避免使用者誤以為該 score 可以比較。

## 3. 整體指標比較

所有指標皆為越高越好。

| 指標 | Reranker 前 | Reranker 後 | 差值 |
|---|---:|---:|---:|
| Hit@1 | 0.4000 | 0.4000 | 0.0000 |
| Macro Precision@1 | 0.4000 | 0.4000 | 0.0000 |
| Macro Recall@1 | 0.3000 | 0.3000 | 0.0000 |
| Macro NDCG@1 | 0.4000 | 0.4000 | 0.0000 |
| Hit@3（primary K） | 0.4000 | 0.4000 | 0.0000 |
| Macro Precision@3 | 0.2000 | 0.2000 | 0.0000 |
| Macro Recall@3 | 0.4000 | 0.4000 | 0.0000 |
| Macro NDCG@3 | 0.3839 | 0.3839 | 0.0000 |
| Hit@5 | 0.8000 | 0.8000 | 0.0000 |
| Macro Precision@5 | 0.2000 | 0.2000 | 0.0000 |
| Macro Recall@5 | 0.8000 | 0.8000 | 0.0000 |
| Macro NDCG@5 | 0.5475 | 0.5562 | +0.0088 |
| MRR | 0.4900 | 0.5000 | +0.0100 |

K=1、3、5 的 depth coverage 都是 1.0000，表示每一筆成功配對的結果都有
足夠深度供這三個 K 評估。這只代表結果清單長度足夠，不代表排序品質良好。

## 4. Reranker 影響

整體判定為「混合」，主要證據如下：

- `query-001` 是成功救援案例。相關文件 `doc-pump-reset` 原本在 Top 3
  之外，reranker 將它移到第 1 名，使 NDCG@3 從 0.0000 升到 1.0000。
- `query-002` 是退步案例。`doc-motor-thermal` 原本排名第 1，reranker
  將它移到第 4，導致它離開 Top 3，使 NDCG@3 從 1.0000 降到 0.0000。
- 上述 +1.0000 與 -1.0000 在 macro average 中互相抵消，因此
  Macro NDCG@3 顯示零差值。
- 共有 1 筆 `reranker_improvement`、1 筆 `reranker_regression`、
  1 筆 `reranker_rescue`，另外 3 筆在 primary K 的 NDCG 數值未變。

Reranker 只重新排列原本的候選集合。`query-001` 的救援表示它把已經存在的
`doc-pump-reset` 從較後位置移到 Top 3，並不是重新檢索到一份新文件。

## 5. 失敗分布

- 評估 Query：5 筆
- Failure case：3 筆
- Failure case rate：60.00%
- `top_k_total_miss`：3 筆（60.00%）
- `persistent_top_k_miss`：2 筆（40.00%）
- `no_relevant_candidate`：1 筆（20.00%）
- `reranker_regression`：1 筆（20.00%）

Failure tag 是 multi-label；同一筆 Query 可以同時具有多個標籤，因此上述
數量與百分比不能相加後當成 Query 總數或總比例。

## 6. 代表性失敗案例

### `query-002`：Motor overheating

- Ground truth：`doc-motor-thermal`
- Before Top 3：`doc-motor-thermal`、`doc-bearing`、`doc-motor-wiring`
- After Top 3：`doc-bearing`、`doc-motor-wiring`、`doc-lubrication`
- 標籤：`top_k_total_miss`、`reranker_regression`
- NDCG@3：1.0000 → 0.0000，差值 -1.0000
- 觀察事實：reranker 把唯一相關文件從第 1 名降到第 4 名。
- 可能原因：reranker 對 motor thermal inspection 的領域用語判斷可能不足；
  這只是待驗證假設。
- 建議檢查：比較五個 candidate chunk 的文字、截斷內容與 reranker 特徵，並
  將此 Query 加入固定 regression set。
- 確認後行動：補強同類 Query-document pair 的訓練或排序規則，預期
  `reranker_regression` 數量下降且 NDCG@3 回升。

### `query-003`：Calibration drift

- Ground truth：`doc-sensor-calibration`
- Before Top 3：`doc-sensor-installation`、`doc-sensor-cleaning`、
  `doc-sensor-wiring`
- After Top 3：與 Before 相同
- 標籤：`no_relevant_candidate`、`top_k_total_miss`、
  `persistent_top_k_miss`、`reranker_unchanged`
- NDCG@3：0.0000 → 0.0000，差值 0.0000
- 觀察事實：完整 candidate pool 中沒有 `doc-sensor-calibration`，reranker
  沒有可以提升的相關候選文件。
- 可能原因：文件可能未被索引、document ID 未對齊、metadata filter 排除它，
  或 first-stage retrieval 沒有召回；目前無法由標籤判定是哪一種。
- 建議檢查：確認文件是否存在於索引、核對 document ID 和過濾條件，再用相同
  Query 進行 first-stage retrieval 測試。
- 確認後行動：修正索引、映射、filter 或 retriever，預期
  `no_relevant_candidate` 比例下降。

### `query-004`：Valve pressure oscillation

- Ground truth：`doc-valve-tuning`
- Before Top 3：`doc-pressure-basics`、`doc-valve-installation`、
  `doc-pipe-sizing`
- After Top 3：`doc-valve-installation`、`doc-pressure-basics`、
  `doc-pipe-sizing`
- 標籤：`top_k_total_miss`、`persistent_top_k_miss`、
  `reranker_unchanged`
- NDCG@3：0.0000 → 0.0000，差值 0.0000
- 觀察事實：相關文件存在於候選集合，但 reranker 後仍在第 4 名，沒有進入
  Top 3。
- 可能原因：Query 與相關 chunk 的語意表達可能不足，或 primary K=3 對此類
  Query 太嚴格；兩者都需要分開驗證。
- 建議檢查：檢視候選 chunk 內容，並比較 reranker 調整與 K=3/K=5 的結果。
- 確認後行動：若排序證據不足，改善 reranker；若產品能接受更多 context，
  再獨立測試提高 K 的品質、延遲與成本影響。

## 7. 改善建議與限制

1. 優先處理 `query-002` 的 reranker regression，因為它破壞了原本成功的
   Top 3 結果。
2. 將 `query-001` 保存為 reranker rescue 的 regression test，避免後續修改
   失去已經取得的改善。
3. 對 `query-003` 先檢查索引、ID mapping、filter 與 first-stage retrieval，
   不要先修改 reranker。
4. 對 `query-004` 分別測試 ranking 改善與 K 調整，避免同時改兩個變因。
5. 修正 `query-006`、`query-999` 的資料配對，以及 `query-001` 的非數值 score，
   再進行下一次完整評估。

本報告只涵蓋 5 筆成功配對 Query，樣本數不足以代表正式環境。Ground truth
可能不完整或過期，binary relevance 也不區分相關程度。結論依賴本次選擇的
K，且 retrieval relevance 不能證明最終生成答案正確、忠實或沒有 hallucination。
本次評估也沒有測量 latency、cost、資料新鮮度或業務結果。
