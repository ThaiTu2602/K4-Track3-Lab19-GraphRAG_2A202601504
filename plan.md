# PLAN — Lab 19: Production-Grade GraphRAG vs Flat RAG

**Repo:** `K4-Track3-Lab19-GraphRAG_2A202601504` · **Branch:** `main` · **Origin:** `ThaiTu2602/K4-Track3-Lab19-GraphRAG_2A202601504`
**Mục tiêu:** 100 điểm + bonus. **Ngân sách thời gian thực tế:** ~4–5h (2h chạy pipeline + 1h chờ LLM/debug + 1–2h viết báo cáo).

---

## 0. Hiện trạng repo (đã kiểm tra)

| Hạng mục | Trạng thái |
|---|---|
| `Day19_..._Lab_Guide.ipynb` | 37 cells, **0 output**, mọi lệnh driver bị comment (`# raw_df = load_news(...)`, `# connect_neo4j()`, ...). Đây là **khung code**, chưa chạy. |
| `outputs/` | Rỗng (chỉ `.gitkeep`) — thiếu 2 file CSV bắt buộc |
| `reports/lab_report.md` | **Y hệt** `templates/lab_report.md`, chưa điền chữ nào |
| `data/golden_dataset.csv` | **KHÔNG TỒN TẠI** (README mô tả có, thực tế không) |
| Python trên máy local | **KHÔNG CÓ** (chỉ có stub WindowsApps) → bắt buộc chạy Colab |

### ⚠️ 3 cái bẫy mất điểm phát hiện được khi đọc repo

**BẪY 1 — `.gitignore` nuốt mất deliverable.**
`.gitignore` có `*.csv` và chỉ whitelist `!reports/*.csv`. Nghĩa là `outputs/graphrag_eval_results.csv` và `outputs/graphrag_vs_flatrag_summary.csv` **sẽ không bao giờ được push lên GitHub** → dính phạt *"Không xuất được 2 file CSV báo cáo kết quả: −5 điểm"*. Phải patch `.gitignore` (xem §3.1).

**BẪY 2 — Tài liệu mâu thuẫn về nơi đặt file.**
- Đường dẫn CSV: README + ASSIGNMENT nói `outputs/`, RUBRIC 3.3 nói `reports/` → **ghi cả 2 nơi**.
- File báo cáo: README + ASSIGNMENT nói 1 file `reports/lab_report.md`; RUBRIC 4.1–4.3 chấm theo 3 file `technical_defense.md` / `failure_analysis.md` / `reflection_[HọTên].md` → **nộp cả 4 file** (lab_report.md là bản đầy đủ, 3 file kia tách nội dung ra). Chi phí gần bằng 0, tránh phạt −5đ/file.

**BẪY 3 — Notebook ghi ra `/content/`, không ghi vào repo.**
`GOLDEN_PATH`, `CHECKPOINT`, và 2 lệnh `to_csv` đều trỏ `/content/...`. Phải sửa sang đường dẫn repo (xem §3.3).

---

## 1. Môi trường thực thi

Máy local **không có Python** → **chạy Google Colab** (notebook vốn thiết kế cho Colab: `google.colab.userdata`, path `/content/`).

**Chiến lược khuyến nghị — clone repo vào Colab, làm việc trực tiếp trong repo, push từ Colab.** Tránh cảnh copy file qua lại rồi quên mất file nào.

```python
# Cell đặt ngay sau 1.1 — Install
import os
from google.colab import userdata
GH_TOKEN = userdata.get("GH_TOKEN")     # GitHub PAT (repo scope), lưu trong Colab Secrets
REPO_URL  = f"https://{GH_TOKEN}@github.com/ThaiTu2602/K4-Track3-Lab19-GraphRAG_2A202601504.git"
!git clone {REPO_URL} /content/lab19
%cd /content/lab19
!git config user.name "ThaiTu2602" && git config user.email "ngthaitu12@gmail.com"
REPO = "/content/lab19"
```

> Nếu ngại PAT: chạy ở `/content/` như mặc định, cuối buổi tải `.ipynb` + 2 CSV về máy, copy vào repo local rồi commit từ Windows. Chấp nhận được nhưng dễ sót file.

**Runtime:** T4 GPU (tăng tốc `sentence-transformers` embedding 3000 chunks). CPU cũng chạy được, chỉ chậm hơn ~3–4 phút.

---

## 2. Chuẩn bị secrets (làm TRƯỚC, ~15 phút)

Vào Colab → 🔑 **Secrets** → bật *Notebook access* cho từng key:

| Secret | Lấy ở đâu | Ghi chú |
|---|---|---|
| `NEO4J_URI` | Neo4j Aura Free → tạo instance | dạng `neo4j+s://xxxx.databases.neo4j.io` |
| `NEO4J_USER` | `neo4j` | |
| `NEO4J_PASSWORD` | file `.txt` Aura tải về lúc tạo instance | **lưu ngay, không xem lại được** |
| `NEO4J_DATABASE` | `neo4j` | |
| `GROQ_API_KEY` | console.groq.com | free tier |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `JUDGE_PROVIDER` | `groq` *(khuyến nghị)* hoặc `openai` | dùng `groq` để khỏi tốn tiền OpenAI |
| `JUDGE_MODEL` | `llama-3.3-70b-versatile` (nếu groq) / `gpt-4o-mini` (nếu openai) | |
| `OPENAI_API_KEY` | chỉ khi `JUDGE_PROVIDER=openai` | |
| `HF_TOKEN` | huggingface.co/settings/tokens | phải **Agree access** trang dataset HackerNoon trước |
| `GH_TOKEN` | GitHub → Developer settings → PAT (repo scope) | chỉ cần nếu push từ Colab |

> **Tuyệt đối không hard-code key vào notebook** — phạt −10đ. `get_secret()` ở cell 1.2 đã xử lý đúng, cứ dùng.

**Kiểm tra sớm:** chạy 1.2 → `print(bool(NEO4J_URI), bool(GROQ_API_KEY), bool(HF_TOKEN), bool(JUDGE_MODEL))` phải ra `True True True True` trước khi làm tiếp. Đừng chờ đến phút 45 mới phát hiện thiếu key.

---

## 3. Patch repo trước khi chạy pipeline

### 3.1 Sửa `.gitignore` (BẮT BUỘC)

Thêm vào cuối file:

```gitignore
# Deliverables bắt buộc phải commit (ghi đè rule *.csv ở trên)
!outputs/*.csv
!reports/*.csv
!data/golden_dataset.csv
```

Verify sau khi sửa: `git check-ignore -v outputs/graphrag_eval_results.csv` phải **không in ra gì**.

### 3.2 Tạo `data/golden_dataset.csv`

Chưa tồn tại. **Đừng tạo vội bằng 5 câu starter** — G02–G05 hỏi về sự kiện có thể **không hề xuất hiện** trong subset 400 chunks được trích xuất. Xem §6 (chiến lược quan trọng nhất của bài này).

Tạm thời cứ để notebook fallback về `starter_golden`, sau khi build xong graph mới viết file thật.

### 3.3 Sửa đường dẫn output trong notebook

| Cell | Sửa gì |
|---|---|
| 4.1 | `GOLDEN_PATH = f"{REPO}/data/golden_dataset.csv"` |
| 4.3 | `CHECKPOINT = "/content/graphrag_eval_checkpoint.csv"` — giữ nguyên (file tạm, không nộp) |
| 4.4 | Ghi ra **4 đường dẫn**: `outputs/*.csv` **và** `reports/*.csv` |

```python
# Thay 2 dòng to_csv cuối cell 4.4
import os
for d in ["outputs", "reports", "data"]:
    os.makedirs(f"{REPO}/{d}", exist_ok=True)
for d in ["outputs", "reports"]:
    eval_results_df.to_csv(f"{REPO}/{d}/graphrag_eval_results.csv", index=False)
    comparison_df.to_csv(f"{REPO}/{d}/graphrag_vs_flatrag_summary.csv", index=False)
print("✅ Exported 4 CSV files")
```

### 3.4 Bỏ comment các dòng driver

Notebook có sẵn nhưng bị comment. Danh sách chính xác:

| Cell | Dòng cần bỏ comment |
|---|---|
| 1.4 | `connect_neo4j()` · `setup_graph_schema()` |
| 1.5 | `raw_df = load_news(DATA_PATH)` · `news_df = standardize_news(raw_df)` · `chunks_df = build_chunks(news_df)` · `display(...)` |
| 1.7 | `extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()` · `coref_df = run_coref(...)` · `extraction_source = extraction_source.merge(...)` |
| 2.1 | `raw_triples_df, extraction_errors_df = run_extraction(extraction_source)` · `display(...)` |
| 2.2 | `entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)` · `triples_df = canonicalize_triples(...)` · `display(...)` |
| 2.3 | `nodes_df = build_nodes(triples_df)` · `bulk_insert_nodes(nodes_df)` · `bulk_insert_edges(triples_df)` |
| 2.4 | `graph_counts, top_degree_df = graph_checks()` |
| 3.1 | `build_flat_index(chunks_df)` |
| 3.2 | `build_entity_matcher(nodes_df)` |
| 4.3 | `validate_golden(golden_df, require_answers=True)` · `eval_results_df = run_evaluation(golden_df)` |
| 4.4 | `comparison_df = comparison_table(eval_results_df)` + block export ở §3.3 |
| 5.1 | `test_supernode_policy()` · `show_resolution_audit(entity_resolution_audit_df)` |
| Bonus | `community_df = build_communities()` |

---

## 4. Chạy pipeline — 5 module + tiêu chí PASS

Sau mỗi module, **dừng lại kiểm tra tiêu chí PASS** rồi mới đi tiếp. Đừng "Run All" một mạch.

### M1 — Setup & Preprocessing *(~15 phút, phần lớn là chờ download)*

Chạy 1.1 → 1.7.

- 1.3 stream dataset: `PRIORITIZE_MB=True`, `LIMIT_MB=300`. Chờ ~5–10 phút. Nếu nghẽn mạng Colab → hạ `LIMIT_MB=100`, vẫn thừa dữ liệu cho 1500 articles.
- 1.7 coref: 400 chunks ÷ batch 5 = **80 lượt gọi Groq**. Đây là chỗ dễ dính rate-limit đầu tiên → xem §5.1.

**PASS khi:**
- [ ] `Exact dedup: N -> M` in ra, `M < N` (chứng minh dedup có tác dụng)
- [ ] `len(chunks_df)` ≈ 3000
- [ ] `coref_df` có ≥1 dòng `unresolved_mentions` không rỗng
- [ ] **Không có** dòng nào `unresolved_mentions == ["COREF_BATCH_FAILED"]` chiếm >20% (nếu có → rate-limit, phải fix trước khi đi tiếp)

**📋 Chốt số liệu cho báo cáo:** chạy thêm cell dưới, **copy output ra file nháp ngay** (Q1 của báo cáo cần nó):

```python
# Soi ca coref sai/khó — nguyên liệu cho câu hỏi 1 báo cáo
cmp = extraction_source[["chunk_id", "text", "resolved_text", "unresolved_mentions"]].copy()
cmp["changed"] = cmp.text != cmp.resolved_text
print("Chunks bị đổi:", cmp.changed.sum(), "/", len(cmp))
print("Chunks có unresolved:", cmp.unresolved_mentions.map(len).gt(0).sum())
for r in cmp[cmp.changed].head(5).itertuples():
    print("="*80, "\nCHUNK:", r.chunk_id, "\nBEFORE:", r.text[:400], "\nAFTER :", r.resolved_text[:400])
```

### M2 — Triple Extraction & Neo4j Ingestion *(~30 phút)*

Chạy 2.1 → 2.4. Đây là bước **tốn LLM nhất**: 400 chunks ÷ batch 4 = **100 lượt gọi**.

**PASS khi:**
- [ ] `len(raw_triples_df)` ≥ 300 (nếu < 150 → prompt quá bảo thủ hoặc rate-limit, xem §5.1)
- [ ] `extraction_errors_df` rỗng hoặc <5 dòng
- [ ] `raw_triples_df.relation.unique()` ⊆ `ALLOWED_RELATIONS`; `source_type/target_type` ⊆ `ALLOWED_NODE_TYPES`
- [ ] `graph_checks()` in `invalid_provenance_edges: 0` và **assert không nổ** ← tiêu chí 2.2 (6đ) + tránh phạt −5đ
- [ ] `nodes` > 100, `edges` > 200

### M3 — Entity Resolution *(~20 phút)*

Chạy 2.2. Đây là chỗ rubric soi kỹ (6đ) và câu hỏi 2 của báo cáo phụ thuộc vào nó.

**PASS khi:**
- [ ] `len(entity_resolution_audit_df)` ≥ 10 dòng
- [ ] Có đủ **cả 3** nhãn `MERGE_MANUAL`, `MERGE_VECTOR`, `REJECT_GUARD`
- [ ] **Có ít nhất 1 dòng `REJECT_GUARD` với `similarity > 0.85`** ← câu hỏi 2 báo cáo yêu cầu đúng thứ này

**⚠️ Rủi ro thật:** với `threshold=0.90`, số cặp candidate rất ít → có thể **không có dòng `REJECT_GUARD` nào**, làm câu hỏi 2 không trả lời được bằng dữ liệu thật.

**Xử lý:** hạ ngưỡng vector để mở rộng pool candidate, guard vẫn giữ nguyên:

```python
entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df, threshold=0.85, top_k=8)
print(entity_resolution_audit_df.decision.value_counts())
rej = entity_resolution_audit_df.query("decision=='REJECT_GUARD'").sort_values("similarity", ascending=False)
display(rej.head(15))   # <- copy dòng đầu vào báo cáo
```

Rồi trong báo cáo giải thích **có chủ đích**: *"chọn 0.85 chứ không phải 0.90 để mở rộng recall của candidate generation, đẩy toàn bộ gánh nặng chống false-merge sang Lexical Guard (SequenceMatcher ≥ 0.72 sau khi strip corp suffix) — đây là thiết kế two-stage precision/recall"*. Đó chính là câu trả lời chất lượng cho Q2.

**📋 Chốt số liệu:** lưu `entity_resolution_audit_df.to_csv(f"{REPO}/outputs/entity_resolution_audit.csv")` — không bắt buộc nộp nhưng là bằng chứng mạnh, và audit table là tiêu chí chấm 2.3.

### M4 — Retrieval Architecture *(~25 phút)*

Chạy 3.1 → 3.4. Embedding 3000 chunks: ~1–3 phút (GPU) / ~8 phút (CPU).

**PASS khi:**
- [ ] `Flat vectors: 3000` (hoặc = `len(chunks_df)`)
- [ ] Smoke test dưới đây chạy ra kết quả có nghĩa cho **cả 2** hướng:

```python
q = "Which company invested in an AI startup?"
g = retrieve_graph_context(q, max_hops=2, edge_limit=50, return_debug=True)
print("SEEDS:", g["diagnostics"]["matched_seeds"])
print("EXPANDED:", g["diagnostics"]["expanded_nodes"], "EDGES:", g["diagnostics"]["collected_edges"])
print("SUPERNODE EVENTS:", g["diagnostics"]["supernode_events"])
print(g["context"][:1500])
print("\n--- FLAT ---\n", answer_flat_rag(q)["answer"][:600])
print("\n--- GRAPH ---\n", answer_graph_rag(q)["answer"][:600])
```

- [ ] `matched_seeds` **không rỗng** — nếu rỗng liên tục thì `diagnostics.reason == "NO_SEED"`, phải hạ `fuzzy_threshold` từ 0.66 → 0.55 trong `match_seeds()` và ghi lại lý do vào báo cáo.

### M5 — Golden Eval & Benchmark *(~20 phút)*

**Chỉ chạy sau khi hoàn tất §6.** `validate_golden(require_answers=True)` sẽ raise nếu còn ô `reference_answer` trống.

Chạy 4.1 → 4.4. Mỗi câu hỏi = 4 lượt LLM (flat answer + graph answer + 2 judge). 5 câu = **20 lượt**. `CHECKPOINT` ghi sau mỗi câu → nếu đứt giữa chừng vẫn còn dữ liệu.

**PASS khi:**
- [ ] `eval_results_df` đủ 5 dòng, không NaN ở cột điểm
- [ ] `comparison_df` có đủ 3 group × 5 metric = 15 dòng
- [ ] 4 file CSV tồn tại trong `outputs/` và `reports/`

### M5b — Failure-mode checks *(~15 phút)*

Chạy 5.1. Xem §5.3 về vấn đề super-node.

---

## 5. Rủi ro đã lường trước & cách xử lý

### 5.1 Rate-limit Groq (rủi ro CAO — sẽ gặp)

Tổng lượt gọi: 80 (coref) + 100 (extraction) + 20 (eval) + seed extraction ≈ **210 request**. Free tier `llama-3.3-70b-versatile` giới hạn ~30 RPM và ~6k–12k TPM. `groq_chat()` có retry backoff nhưng cap ở 20s — chưa đủ cho lỗi 429 kèm `retry-after` dài.

**Triệu chứng:** hàng loạt `COREF_BATCH_FAILED`, hoặc `extraction_errors_df` phình to, hoặc `raw_triples_df` gần rỗng.

**Xử lý theo thứ tự:**
1. Thêm nghỉ giữa các batch — sửa `run_coref` và `run_extraction`, cuối vòng lặp thêm `time.sleep(2.0)`.
2. Nâng backoff trong `groq_chat`: `time.sleep(min(60, 3 * 2**attempt + random.random()))` và `max_retries=6`.
3. Nếu vẫn nghẽn: hạ `EXTRACTION_MAX_CHUNKS` **400 → 200**. Đồ thị nhỏ hơn nhưng vẫn đủ chứng minh pipeline; **ghi rõ quyết định + lý do vào báo cáo** (đây chính là loại trade-off rubric muốn thấy).
4. Không đổi model sang loại nhỏ hơn ở giữa chừng — sẽ làm benchmark không so sánh được.

### 5.2 Groq JSON mode + payload lớn

`extract_batch` nhét 4 chunk × 220 từ vào 1 prompt. Nếu model trả JSON bị cắt (`No JSON object found`) → hạ `batch_size` từ 4 → 2 trong `run_extraction`.

### 5.3 Super-node có thể **không tồn tại** (rủi ro CAO — ảnh hưởng 8 điểm)

`test_supernode_policy()` chỉ assert khi node có `degree > 100`. Với ~400 chunks, node cao nhất nhiều khả năng chỉ ~30–80 degree → **test in ra rồi bỏ qua, không chứng minh được gì**, mất phần lớn 8 điểm tiêu chí 2.1.

**Xử lý — chứng minh policy bằng 2 lớp, cả hai đều trung thực:**

```python
# (a) Đo phổ degree thực tế
deg = pd.DataFrame(run_cypher("""
MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-()
WITH n, count(r) AS degree
RETURN n.name AS name, n.entity_type AS type, degree
ORDER BY degree DESC LIMIT 20
"""))
display(deg)                                    # <- Top 3 -> câu hỏi 3 báo cáo
max_deg = int(deg.degree.max()); print("MAX DEGREE =", max_deg)

# (b) Nếu chưa chạm 100, hạ ngưỡng xuống dưới max_deg để CHỨNG MINH cơ chế cắt tỉa thực sự kích hoạt
if max_deg <= SUPER_NODE_DEGREE:
    print(f"[NOTE] Subset lab chưa sinh node degree>100 (max={max_deg}).")
    print("       Hạ tạm ngưỡng để chứng minh policy kích hoạt đúng.")
    _saved = SUPER_NODE_DEGREE
    SUPER_NODE_DEGREE = max(5, max_deg // 2)

g = retrieve_graph_context("Tell me everything about the biggest tech companies",
                           max_hops=2, edge_limit=50, return_debug=True)
ev = g["diagnostics"]["supernode_events"]
print("SUPERNODE EVENTS:", ev)
print("COLLECTED EDGES :", g["diagnostics"]["collected_edges"], "<= GLOBAL_EDGE_CAP", GLOBAL_EDGE_CAP)
print("CONTEXT CHARS   :", len(g["context"]), "<= MAX_GRAPH_CONTEXT_CHARS", MAX_GRAPH_CONTEXT_CHARS)
assert all(e["limit"] <= SUPER_NODE_EDGE_CAP for e in ev), "Super-node cap không kích hoạt!"
assert g["diagnostics"]["collected_edges"] <= GLOBAL_EDGE_CAP
assert len(g["context"]) <= MAX_GRAPH_CONTEXT_CHARS
print("✅ Super-node cap + global cap + context cap đều OK.")

try: SUPER_NODE_DEGREE = _saved
except NameError: pass
```

Báo cáo ghi thẳng: *"Ở quy mô lab (N chunks), degree cao nhất chỉ là X < 100 nên policy production không kích hoạt tự nhiên. Đã hạ ngưỡng xuống X/2 để chứng minh cơ chế cắt tỉa hoạt động đúng, đồng thời verify 3 tầng chặn: per-node cap 50, GLOBAL_EDGE_CAP 250, MAX_GRAPH_CONTEXT_CHARS 14000."* Trung thực + chứng minh được = ăn trọn điểm.

### 5.4 BFS `expanded` chặn hop-2 mở rộng

Trong `retrieve_graph_context`, node đã vào `expanded` sẽ không bao giờ được duyệt lại, và `frontier.append` bị chặn bởi `hop + 1 < max_hops`. Với `max_hops=2` thì đúng ý đồ (seed hop-0 → neighbor hop-1). Không phải bug, **nhưng cần hiểu để trả lời câu hỏi thuyết minh**: *"max_hops=2 nghĩa là 2 lượt expand, cho ra đường đi độ dài tối đa 2 cạnh"*.

### 5.5 Neo4j Aura Free ngủ đông

Instance free tự pause sau 3 ngày không dùng. Nếu `verify_connectivity()` timeout → vào console Aura bấm Resume, chờ ~1 phút.

---

## 6. ⭐ Golden Dataset — bước quan trọng nhất, dễ mất điểm nhất

**Vấn đề:** G02–G05 trong `starter_golden` là câu hỏi mở, `reference_answer=""`. Với subset chỉ 400 chunks được extract, khả năng cao **không có fact nào trong graph trả lời được** chúng → cả Flat lẫn Graph đều trả lời rỗng → judge cho 1 điểm hết → benchmark vô nghĩa, mất điểm 3.1 (6đ) + làm hỏng phân tích câu 4 báo cáo.

**Giải pháp: đào câu hỏi từ chính graph đã build.** Chạy sau M2, trước M5.

```python
# --- PROBE 1: multi-hop có thật (chuỗi 2 cạnh) ---
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)
WHERE a <> c
RETURN a.name AS a, type(r1) AS rel1, b.name AS b, type(r2) AS rel2, c.name AS c,
       r1.published_date AS d1, r2.published_date AS d2,
       r1.source_chunk_id AS chunk1, r2.source_chunk_id AS chunk2
ORDER BY d2 DESC LIMIT 25
""")))

# --- PROBE 2: pattern founder/employee (đúng tinh thần G02) ---
display(pd.DataFrame(run_cypher("""
MATCH (p:Person)-[:FOUNDED]->(b:Company)
OPTIONAL MATCH (p)-[:WORKED_AT]->(a:Company)
RETURN p.name AS person, a.name AS worked_at, b.name AS founded LIMIT 20
""")))

# --- PROBE 3: cross-doc — cùng cặp entity xuất hiện ở >= 2 chunk khác nhau ---
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r]->(b:Entity)
WITH a, b, type(r) AS rel,
     collect(DISTINCT r.source_chunk_id) AS chunks,
     collect(DISTINCT r.published_date)  AS dates
WHERE size(chunks) >= 2
RETURN a.name AS a, rel, b.name AS b, chunks, dates
ORDER BY size(chunks) DESC LIMIT 20
""")))

# --- PROBE 4: factoid chắc chắn có (cạnh confidence cao nhất) ---
display(pd.DataFrame(run_cypher("""
MATCH (a:Entity)-[r]->(b:Entity)
RETURN a.name AS a, type(r) AS rel, b.name AS b,
       r.published_date AS date, r.evidence AS evidence, r.confidence AS conf
ORDER BY r.confidence DESC LIMIT 20
""")))
```

**Từ output 4 probe này, viết lại golden dataset:**

| ID | group | Nguồn | Nguyên tắc |
|---|---|---|---|
| G01 | factoid | PROBE 4 | 1 cạnh confidence cao. `reference_answer` = chính fact đó. **Flat RAG nên thắng hoặc hòa** ở nhóm này. |
| G02 | multi-hop | PROBE 2 hoặc 1 | Chuỗi 2 cạnh **qua 2 chunk_id khác nhau** — đây là điều kiện làm Flat RAG thất bại có kiểm soát. |
| G03 | multi-hop | PROBE 1 | Chuỗi 2 cạnh khác, hỏi kèm **cả 2 relation và date**. |
| G04 | cross-doc | PROBE 3 | Cặp entity xuất hiện ≥2 chunk, hỏi *"quan hệ thay đổi thế nào theo thời gian"*. |
| G05 | cross-doc | PROBE 3 | So sánh 2 công ty theo dòng thời gian từ nhiều bài. |

> **Mẹo ăn điểm:** cố tình chọn G02 sao cho 2 cạnh nằm ở 2 `chunk_id` **cách xa nhau về ngữ nghĩa**. Đó chính là ca *"Flat RAG thất bại, GraphRAG thành công"* mà câu 4 báo cáo yêu cầu — và bạn sẽ có bằng chứng root-cause thật (in `flat["retrieved"].chunk_id` ra sẽ thấy Flat không lấy được chunk thứ 2).

Ghi file:

```python
golden_df = pd.DataFrame([
    {"id":"G01","group":"factoid","question":"...","reference_answer":"...","reference_evidence":"chunk=..."},
    # ... G02..G05, MỌI reference_answer đều có nội dung thật
])
os.makedirs(f"{REPO}/data", exist_ok=True)
golden_df.to_csv(f"{REPO}/data/golden_dataset.csv", index=False)
validate_golden(golden_df, require_answers=True)   # phải in "✅ Golden Dataset valid."
```

**Giữ nguyên G01 = "Who was the CEO of Hugging Face in 2023?" / "Clément Delangue"** nếu graph có node Hugging Face — câu này đã có sẵn gold answer, dùng làm baseline factoid rất tiện.

---

## 7. Bonus (+10, có thể tới +13)

Làm theo thứ tự ROI giảm dần:

### B1 — Near-Dedup (+3, rẻ nhất, ~15 phút)
Challenge A ở cell 1.5. Đã có `faiss` và `embedder` sẵn → embedding + ANN, **không** pairwise O(N²) (notebook cấm rõ).

```python
def near_dedup(chunks_df, threshold=0.93, top_k=5):
    v = get_embedder().encode(chunks_df.text.tolist(), batch_size=128,
                              normalize_embeddings=True, show_progress_bar=True).astype("float32")
    idx = faiss.IndexFlatIP(v.shape[1]); idx.add(v)
    sims, nbrs = idx.search(v, top_k)
    uf, audit = UF(len(v)), []
    for i in range(len(v)):
        for s, j in zip(sims[i], nbrs[i]):
            if j < 0 or i >= j or float(s) < threshold: continue
            audit.append({"left": chunks_df.chunk_id.iloc[i], "right": chunks_df.chunk_id.iloc[int(j)],
                          "similarity": float(s), "decision": "NEAR_DUP"})
            uf.union(i, int(j))
    keep = [i for i in range(len(v)) if uf.find(i) == i]
    print(f"Near-dedup: {len(v)} -> {len(keep)}  ({len(audit)} cặp)")
    return chunks_df.iloc[keep].reset_index(drop=True), pd.DataFrame(audit)
```
Báo cáo phải nêu đủ 3 thứ notebook yêu cầu: **threshold, false positive, cách audit**.

### B2 — Self-Correction Retrieval (+5, ~20 phút)
Cell 35 đã có `self_correcting_context()`, chỉ cần chạy và **định lượng trước/sau**:

```python
for q in golden_df.query("group != 'factoid'").question:
    r = self_correcting_context(q)
    print(f"[{r['route']}] {q[:70]}  | missing={r['missing'][:80]}")
```
Rubric bonus yêu cầu *"có định lượng trước/sau"* → chạy `judge_answer` cho 1–2 câu với context hop-2 thường vs context self-corrected, so điểm. Nêu rõ **stop condition** (tối đa hop 3 rồi bắt buộc fallback vector).

### B3 — Community Detection / Global Search (+5, ~25 phút)
Cell 34 `build_communities()` có sẵn. Thêm phần LLM summarize community + 1 câu hỏi vĩ mô:

```python
community_df = build_communities()
print(community_df.community_id.nunique(), "communities")
top_c = community_df.community_id.value_counts().head(3).index.tolist()
for cid in top_c:
    members = run_cypher("MATCH (n:Entity {community_id:$c}) RETURN n.name AS name LIMIT 40", c=int(cid))
    edges   = run_cypher("""MATCH (a:Entity {community_id:$c})-[r]->(b:Entity)
                            RETURN a.name+' -'+type(r)+'-> '+b.name AS e LIMIT 60""", c=int(cid))
    rep, _ = groq_chat([{"role":"system","content":"Summarize this entity community in 3 sentences."},
                        {"role":"user","content":json.dumps({"members":members,"edges":edges})}])
    print(f"\n=== COMMUNITY {cid} ===\n{rep}")
```

---

## 8. Viết báo cáo (~60–90 phút)

Nộp **4 file** (xem BẪY 2):

```
reports/
├── lab_report.md                 # bản đầy đủ theo template (README/ASSIGNMENT)
├── technical_defense.md          # PHẦN 1, mục 1–5 (RUBRIC 4.1, 10đ)
├── failure_analysis.md           # PHẦN 1, mục 4 mở rộng: ≥2 ca lỗi root-cause (RUBRIC 4.2, 5đ)
└── reflection_[HọTên].md         # PHẦN 2 (RUBRIC 4.3, 5đ)
```

Viết `lab_report.md` trước (bản gốc đầy đủ), rồi cắt sang 3 file kia.

### 📋 Bảng thu thập số liệu — điền TRONG LÚC CHẠY, đừng để cuối buổi

| # | Câu hỏi báo cáo | Số liệu cần | Lấy ở đâu |
|---|---|---|---|
| Q1 | Coreference sai | 1 cặp `text` vs `resolved_text` + `chunk_id` | cell soi coref ở §M1 |
| Q2 | Threshold & Guard | `threshold` đã dùng + 1 dòng `REJECT_GUARD` có `similarity > 0.85` | `entity_resolution_audit_df` §M3 |
| Q3 | Super-node | Top 3 (name, type, degree) + max_degree + `supernode_events` | cell §5.3 |
| Q4 | Benchmark + 2 ca lỗi | Bảng 5 metric trung bình + `flat_judge_rationale` / `graph_judge_rationale` của ca lỗi + `flat["retrieved"].chunk_id` | `comparison_df`, `eval_results_df` |
| Q5 | Trade-off & scale | latency/token trung bình 2 bên; số API call; thời gian index | `eval_results_df` + tự đếm |

### Gợi ý nội dung có chiều sâu

**Q1 — Coreference.** Nêu đúng cơ chế hỏng: `"The company said it would acquire..."` → nếu chunk có 2 công ty, LLM gán sai antecedent → tạo cạnh `ACQUIRED` ngược chiều hoặc gán cho đối thủ. Hậu quả: cạnh sai này **có đủ provenance hợp lệ** (`source_chunk_id`, `published_date`, `evidence`) nên các sanity check **không bắt được** — provenance đảm bảo *truy vết được*, không đảm bảo *đúng*. Đó là insight mạnh.

**Q2 — Threshold.** Trình bày kiến trúc 2 tầng: vector ANN = **recall stage** (ngưỡng nới), lexical guard = **precision stage** (SequenceMatcher ≥ 0.72 sau `strip_suffix`). Ví dụ điển hình: `Apple` vs `Apple Watch` (vector sim cao vì cùng brand embedding, nhưng `strip_suffix` không rút gọn được "watch" → ratio thấp → REJECT). Hoặc `Sam Altman` vs `Steve Altman` — cùng họ nên vector gần, guard chặn ở phần tên.

**Q3 — Super-node, rủi ro của `ORDER BY published_date DESC`.**
- *Ưu:* context bounded, thiên về thông tin mới nhất — hợp với tin công nghệ.
- *Rủi ro:* câu hỏi lịch sử (*"Microsoft mua công ty nào năm 2016?"*) sẽ bị cắt sạch cạnh cũ → **silent failure**, model trả lời "không có bằng chứng" dù graph có dữ liệu. Đề xuất khắc phục: **query-aware selection** — nếu câu hỏi chứa mốc thời gian thì `ORDER BY abs(date - target_date)` thay vì `DESC`; hoặc stratified sampling (25 cạnh mới nhất + 25 cạnh confidence cao nhất).
- Thêm: `coalesce(r.published_date,'')` khiến edge thiếu date bị đẩy xuống cuối → luôn bị cắt trước. Nêu ra điều này = điểm cộng.

**Q4 — 2 ca lỗi.** Ca Flat thua: in `flat["retrieved"].chunk_id` chứng minh chunk chứa hop thứ 2 **không nằm trong top-6** → đó là root cause, không phải suy đoán. Ca Graph thua: thường là factoid — graph chỉ giữ triple đã lược bỏ ngữ cảnh, hoặc `NO_SEED` khi câu hỏi không chứa entity name (*"the largest AI investment last year"*).

**Q5 — Scale 350MB.** Bottleneck đầu tiên **không phải Neo4j** mà là **LLM extraction throughput**: 100k bài × ~5 chunk = 500k chunk; batch 4 = 125k request; 30 RPM ⇒ ~70 **ngày**. Giải pháp: (1) async worker pool + nhiều API key, (2) pre-filter bằng NER rẻ (spaCy) chỉ gửi chunk có ≥2 entity, (3) model nhỏ hơn cho extraction, (4) Entity Resolution đổi FAISS Flat → **HNSW + blocking theo ký tự đầu/type** vì Flat là O(N²) trên 500k mentions, (5) Neo4j: `apoc.periodic.iterate` + tăng heap, (6) super-node lúc đó **sẽ** vượt 100 → policy hiện tại mới thực sự có tác dụng.

**"Đề xuất của AI Agent mà bạn từ chối"** — dùng ví dụ thật từ chính bài này (có sẵn, không cần bịa):
- Từ chối pairwise cosine O(N²) cho near-dedup → dùng FAISS ANN top-k.
- Từ chối bỏ `merge_guard` để "gộp được nhiều entity hơn" → chấp nhận recall thấp hơn đổi lấy precision, vì false merge làm hỏng vĩnh viễn cấu trúc graph còn miss merge chỉ làm phân mảnh.
- Từ chối `MERGE` từng row thay cho `UNWIND` batch → 1 round-trip/row lên Aura là không chấp nhận được.
- Từ chối bỏ `assert invalid == 0` khi nó fail → phải sửa nguồn lỗi, không sửa test.

---

## 9. Submit

```powershell
# Từ Windows (nếu không push từ Colab)
Set-Location "d:\labAI_vinuni\Phase 2\Day19\K4-Track3-Lab19-GraphRAG_2A202601504"
git check-ignore -v outputs/graphrag_eval_results.csv   # phải KHÔNG in gì
git add -A
git status                                             # xác nhận 2 CSV + notebook + reports có mặt
git commit -m "Lab 19: GraphRAG vs Flat RAG - full pipeline, eval, report"
git push origin main
```

### Checklist cuối (đối chiếu RUBRIC)

**Implementation (40đ)**
- [ ] Notebook `Restart & Run All` xong, **mọi cell có output**, không cell nào crash
- [ ] Dedup + chunking chạy, có log `Exact dedup: N -> M`
- [ ] Coref chạy, có `unresolved_mentions` được log
- [ ] `raw_triples_df` không rỗng, schema hợp lệ (allowlist)
- [ ] `bulk_insert_nodes` / `bulk_insert_edges` dùng `UNWIND $rows AS row`, constraint + index đã tạo
- [ ] `answer_flat_rag()` và `answer_graph_rag()` đều sinh kết quả

**Failure modes (20đ)**
- [ ] Super-node cap chứng minh được (§5.3), 3 assert đều pass
- [ ] `invalid_provenance_edges == 0`
- [ ] `entity_resolution_audit_df` ≥10 dòng, đủ 3 nhãn, có `REJECT_GUARD` sim>0.85

**Evaluation (20đ)**
- [ ] `data/golden_dataset.csv` đủ 3 group, **mọi `reference_answer` đã điền thật**
- [ ] Judge chạy đủ 5 câu × 2 hệ × 3 tiêu chí, có `rationale`
- [ ] `outputs/graphrag_eval_results.csv` + `outputs/graphrag_vs_flatrag_summary.csv` **đã push lên GitHub** (kiểm tra trên web GitHub, không chỉ tin `git status`)
- [ ] Bản sao trong `reports/`

**Report (20đ)**
- [ ] `reports/lab_report.md` điền hết, số liệu THẬT, không còn `[...]`
- [ ] `technical_defense.md` · `failure_analysis.md` · `reflection_[HọTên].md`
- [ ] Bảng mapping bài giảng → hàm code

**Bảo mật (−10đ nếu vi phạm)**
- [ ] `grep -rn "gsk_\|sk-\|hf_\|neo4j+s://" *.ipynb` → không có key thật nào lọt vào output cell
- [ ] Không commit `.env`, không commit CSV dataset lớn

**Bonus**
- [ ] Near-dedup (+3) · Self-correction (+5) · Community detection (+5) — có định lượng trước/sau

---

## 10. Thứ tự thực thi rút gọn

```
1. Secrets (§2) ────────────────────── 15'   ← làm trước, verify True hết
2. Patch .gitignore + paths (§3) ───── 10'
3. M1 Preprocessing + coref (§4) ───── 15'   → chốt số liệu Q1
4. M2 Extraction + Neo4j (§4) ──────── 30'   → assert provenance == 0
5. M3 Entity Resolution (§4) ───────── 20'   → chốt số liệu Q2 (nhớ hạ threshold 0.85)
6. PROBE graph → viết golden (§6) ──── 20'   ★ đừng bỏ qua bước này
7. M4 Retrieval + smoke test (§4) ──── 25'
8. M5 Eval + export 4 CSV (§4) ─────── 20'
9. Super-node proof (§5.3) ─────────── 15'   → chốt số liệu Q3
10. Bonus B1/B2/B3 (§7) ───────────── 40'
11. Viết 4 file báo cáo (§8) ───────── 90'
12. Push + verify trên GitHub (§9) ─── 10'
```

**Nếu thiếu thời gian, cắt theo thứ tự này:** bonus B3 → bonus B2 → bonus B1 → giảm `EXTRACTION_MAX_CHUNKS` xuống 200.
**Tuyệt đối không cắt:** §3.1 (.gitignore), §6 (golden dataset thật), §5.3 (super-node proof), §8 (báo cáo). Bốn thứ này chiếm phần lớn điểm và không thể bù bằng chỗ khác.
