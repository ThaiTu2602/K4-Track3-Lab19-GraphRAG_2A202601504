# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thái Tú
**MSSV:** 2A202601504
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19 – 2026-08-20

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

**Tình huống cụ thể:** Chunk `77f7018d6b3f8b793992::c0000`, tiêu đề *"There's No Substitute for Chinese Drones (and That's a Problem)"*.

```
BEFORE: (and That's a Problem)
AFTER : (and The lack of substitute for Chinese Drones is a Problem)
```

**Hiện tượng:** Đây không phải là lỗi "nhầm sang thực thể khác trong câu trước" như dạng lỗi phổ biến nhất, mà là một dạng lỗi tinh vi hơn: thay vì chỉ **thay thế nguyên văn** đại từ `That` bằng antecedent của nó, model đã **diễn giải lại (paraphrase)** toàn bộ cụm từ thành `The lack of substitute for Chinese Drones is a Problem` — vi phạm trực tiếp ràng buộc *"Never invent facts... Preserve dates, numbers, tickers and product names"* trong system prompt, vốn ngụ ý chỉ nên làm phép thay thế (substitution), không phải viết lại câu.

**Hậu quả đối với Knowledge Graph:** Nếu chunk này lọt vào bước NER/RE extraction (Module 2), cụm từ bị "hallucinate" này có nguy cơ:
- Bị LLM extraction hiểu nhầm thành một entity `Technology` giả (ví dụ node tên `"the lack of substitute for Chinese Drones"`), làm nhiễu đồ thị bằng một node vô nghĩa không tồn tại trong thực tế; hoặc
- Bị bỏ qua hoàn toàn (an toàn hơn nhưng gây mất thông tin nếu câu gốc thực sự có ý nghĩa).

Cả hai đều là hậu quả tiêu cực: **hoặc làm nhiễu graph bằng node giả, hoặc làm mất recall** — chứng minh rằng conservative coreference resolution vẫn cần thêm một lớp kiểm tra hậu kỳ (validate rằng câu sau khi resolve vẫn là một *substring hợp lệ* của câu gốc cộng antecedent, không phải văn bản hoàn toàn mới).

**Ca lỗi phụ (cùng họ vấn đề):** Chunk `fa44e0ddb529ef9fc5c9::c0000` — entity đúng (`Dar Al-Handasah Consultants – Dar`) nhưng thay thế máy móc, lặp lại nguyên cụm tên đầy đủ + tên viết tắt 3 lần trong 2 câu, tạo văn bản ngữ pháp gượng — không sai entity nhưng thiếu bước làm sạch hậu kỳ.

**Đối chiếu — cơ chế hoạt động đúng:**
- Chunk `d38fc817b7dc3eeeb535::c0000` (trang chặn bot/captcha, không phải tin thật) → `unresolved_mentions: ['this', 'you', 'it']`. Model đúng đắn từ chối suy diễn khi không có antecedent rõ ràng.
- Chunk `69f2aecea98a9294feb6::c0000`: `"He issued this directive"` → `"Governor Mohammed Umar Bago issued this directive"` — thay thế sạch, chính xác, không mơ hồ.

Trên tổng 400 chunk gửi trích xuất: **202 chunk bị thay đổi**, **77 chunk có `unresolved_mentions`** (từ chối suy diễn), **0 batch lỗi hệ thống**.

---

### 2. Entity Resolution Threshold & Lexical Guard

**Ngưỡng cosine similarity:** `threshold = 0.85` (hạ từ mặc định `0.90` — mở rộng recall của candidate generation bằng vector ANN, đẩy gánh nặng chống false-merge sang Lexical Guard tầng sau — kiến trúc two-stage precision/recall).

**Kết quả thực tế** (`entity_resolution_audit.csv`, quy mô 400 chunk / 121 triple): chỉ có **4 cặp** vượt ngưỡng candidate generation, cả 4 đều `MERGE_VECTOR` — không có cặp nào bị `REJECT_GUARD` ở quy mô lab này:

| Loại | Thực thể A | Thực thể B | Similarity | Quyết định |
|---|---|---|---|---|
| Company | Reliance Industries Ltd | Reliance Industries | 0.945 | MERGE_VECTOR |
| Company | Activision Blizzard | Activision Blizzard Inc. | 0.918 | MERGE_VECTOR |
| Company | Airbnb | Airbnb Inc. | 0.875 | MERGE_VECTOR |
| Technology | Chandrayaan-I | Chandrayaan-III | 0.858 | MERGE_VECTOR |

**Phát hiện đáng chú ý hơn một ca REJECT_GUARD "sạch":** cặp cuối — `Chandrayaan-I` vs `Chandrayaan-III` (similarity 0.858) — là một **false merge tiềm ẩn**, không phải guard hoạt động đúng. Đây là 2 sứ mệnh không gian **khác nhau** của Ấn Độ, không phải cùng một thực thể viết khác dạng. Lexical Guard trong lab dùng `strip_suffix` (bỏ hậu tố doanh nghiệp như Inc/Corp/Ltd) rồi so `SequenceMatcher.ratio() >= 0.72` — cơ chế này bắt hậu tố *doanh nghiệp*, không phải hậu tố *số thứ tự*. Vì hai tên chỉ khác nhau đúng 1 ký tự số La Mã ở cuối, `SequenceMatcher` cho ratio rất cao → Guard không chặn được, gộp nhầm 2 thực thể khác nhau thành 1 node.

**Bài học:** Lexical Guard xử lý tốt lớp lỗi "cùng thực thể, viết khác dạng" nhưng **chưa xử lý được lớp lỗi "khác thực thể, tên gần giống theo số thứ tự/phiên bản"**. Đề xuất: thêm rule — nếu 2 tên chỉ khác nhau ở token số/số La Mã cuối chuỗi, tự động hạ điểm similarity hoặc bắt buộc review thủ công.

---

### 3. Đồ thị & Super-node Mitigation

**Bằng chứng thực tế:** cột `graph_supernode_events` trong `graphrag_eval_results.csv` bằng **0 trên toàn bộ 25/25 câu hỏi** — ở quy mô lab (400 chunk → 121 triple), không entity nào chạm ngưỡng `SUPER_NODE_DEGREE=100`, cơ chế cắt tỉa chưa từng phải kích hoạt.

**Top 3 Super-nodes:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | *(đang chờ output `top_degree_df` / `test_supernode_policy()`)* | | |
| 2 | | | |
| 3 | | | |

**Ưu điểm & Rủi ro của Temporal Mitigation (lấy 50 cạnh mới nhất theo `published_date DESC`):**
- *Ưu điểm:* giới hạn context bounded, tránh nổ token khi 1 entity có hàng trăm cạnh; ưu tiên thông tin cập nhật — hợp với tin tức công nghệ thay đổi nhanh.
- *Rủi ro:* câu hỏi về sự kiện lịch sử xa có thể bị cắt mất cạnh liên quan nếu node đó là super-node — silent failure. `coalesce(r.published_date,'')` đẩy cạnh thiếu ngày xuống cuối, luôn bị cắt trước tiên khi vượt cap.
- *Rủi ro thực tế quan sát được trong benchmark:* **không phải** do cơ chế cắt tỉa (đã chứng minh chưa từng kích hoạt), mà do **seed-entity matching sai** ở bước trước — xem Ca 2 ở mục 4. Ở quy mô production, cần giám sát song song cả seed-matching lẫn super-node cap, không chỉ riêng cơ chế cắt tỉa.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Golden dataset: **25 câu hỏi thật** (2 factoid, 12 multi-hop, 11 cross-doc), lấy từ repo gốc giảng viên, neo vào 5000 dòng đầu `hackernoon_subset.csv`.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình có trọng số n=25):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|-------------------|----------|----------|---|---------------------|
| **Comprehensiveness (1–5)** | 1.84 | 1.80 | -0.04 | Gần như ngang nhau ở quy mô lab |
| **Faithfulness (1–5)** | 2.08 | 1.92 | -0.16 | Flat nhỉnh hơn — do giới hạn coverage của extraction budget (xem dưới) |
| **Multi-hop Reasoning (1–5)** | 1.84 | 1.80 | -0.04 | Gần như ngang nhau tổng thể; GraphRAG thắng rõ trong nhóm cross-doc (1.82 vs 1.73) |
| **Latency trung bình (s)** | 1.96 | 1.99 | +0.03 (~1.5%) | Không đáng kể dù GraphRAG có thêm bước seed-match + Cypher query |
| **Token usage trung bình** | 637.6 | 563.6 | -74.0 (**-11.6%**) | GraphRAG rẻ hơn — context dạng triple cô đọng hơn đoạn văn thô |

**Nhận xét trung thực:** GraphRAG không thắng đều — điểm mạnh chỉ lộ rõ ở đúng loại câu hỏi nó được thiết kế cho (nối 2 sự kiện rời rạc về cùng 1 entity), không phải cải thiện trung bình mọi câu hỏi. Nguyên nhân chính: ngân sách trích xuất giới hạn (400/5000 chunk) khiến nhiều bằng chứng chưa kịp vào graph dưới dạng triple, trong khi Flat RAG index xây trên toàn bộ chunk nên vẫn "chạm" nhiều đoạn văn thô hơn.

#### Phân tích 2 Ca lỗi Điển hình:

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công): `G5000-44`**
   - *Câu hỏi:* "What two distinct partner ecosystems connect L&T Technology Services to advanced infrastructure in 2023: one for urban-rail 5G and one for OT security?"
   - *Điểm:* Flat RAG (2,2,2) vs GraphRAG (**4,5,4**)
   - *Tại sao Flat RAG thất bại:* vector search single-hop chỉ khớp 1 trong 2 chunk liên quan (Thales/5G), tự nhận thiếu dữ liệu OT security.
   - *GraphRAG giải quyết thế nào:* graph traversal từ node "L&T Technology Services" nối ra 2 cạnh riêng biệt (2 bài báo khác nguồn — Thales cho 5G, Palo Alto Networks cho OT security), trả lời đúng cả hai.

2. **Ca lỗi GraphRAG thất bại (hallucination): `G5000-26`**
   - *Câu hỏi:* "What external technology provider is named inside Amazon's July AI-service expansion...?" (Reference: Cohere)
   - *Điểm:* Flat RAG (1,1,1) vs GraphRAG (1,1,1) — cùng điểm nhưng khác bản chất: Flat trả lời "không đủ bằng chứng" (an toàn), GraphRAG khẳng định sai "Synopsys" (hallucination tự tin).
   - *Nguyên nhân (đã xác minh, không suy đoán):* `graph_supernode_events=0` loại trừ nguyên nhân super-node cap. Root cause thật là **seed-entity matching sai** (`match_seeds()`) — cùng 1 chunk Synopsys lặp lại bất thường ở 3 câu hỏi không liên quan (`G5000-26`, `G5000-30`, `G5000-49`), cho thấy embedding của chunk này là một "attractor" chung chung bị match nhầm bất kể entity thật trong câu hỏi.
   - *Đề xuất khắc phục:* ưu tiên exact-match tên entity trước khi fallback vector similarity; thêm ngưỡng phân tách giữa candidate tốt nhất và thứ nhì, trả lời "không đủ bằng chứng" khi ambiguous thay vì chọn liều.

   *(Phân tích chi tiết hơn, gồm ca lỗi thứ 3 — cả 2 phương pháp cùng thất bại do giới hạn coverage dữ liệu — xem `reports/failure_analysis.md`.)*

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

**Đánh đổi Quality vs Cost vs Latency (số liệu thật, 25 câu):** GraphRAG rẻ hơn về token (-11.6%) nhưng không nhanh hơn (+1.5% latency) và không vượt trội về điểm chất lượng ở quy mô lab (400 chunk/121 triple) — chi phí indexing/vận hành (Neo4j, entity resolution, seed matching) chưa được "khấu hao" vì tập dữ liệu còn nhỏ. Điểm mạnh thật sự chỉ lộ ra ở case cụ thể cần nối nhiều nguồn về cùng 1 entity (xem `G5000-44` ở mục 4) — tức GraphRAG "lời" khi tỉ lệ câu hỏi cross-doc/multi-hop đủ cao trong tập dữ liệu thật.

**Quyết định với AI Coding Agent (3 ví dụ thật):**
1. Khi coreference chậm bất thường (~100s/batch), Agent đề xuất chuyển sang `ThreadPoolExecutor` ngay — **từ chối** vì chưa xác định nguyên nhân gốc (hoá ra là model bị Groq ngừng hỗ trợ, không phải do gọi tuần tự). Về sau thử concurrency thật ở bước extraction, dữ liệu xác nhận quyết định trì hoãn ban đầu đúng: concurrency làm thông lượng **giảm** do rate-limit contention.
2. Agent ban đầu đề xuất đổi Judge sang OpenAI/OpenRouter — **từ chối lúc đầu** (giữ kiến trúc đơn giản, 1 provider). Quyết định **đảo ngược có căn cứ** khi Judge model trên Groq thực sự lỗi (`json_validate_failed`, hết ngân sách token trước khi sinh xong JSON) giữa lần chạy 25 câu — bài học: sẵn sàng đảo ngược quyết định khi có bằng chứng lỗi thật, không cố chấp giữ nguyên vì lý do kiến trúc.
3. `.gitignore` có nguy cơ chặn nhầm CSV bắt buộc nộp — Agent đề xuất bỏ hẳn rule chặn `*.csv`; chọn phương án an toàn hơn: thêm allow-list cụ thể (`!outputs/*.csv`) thay vì xoá rule gốc, tránh commit nhầm file dataset 300MB.

**Giải pháp scale lên 350MB (~100,000 bài báo):**

Bottleneck đầu tiên **không phải Neo4j** mà là **thông lượng LLM extraction** — ước tính vài trăm nghìn chunk, ở rate-limit free-tier chạy tuần tự sẽ mất hàng chục ngày.

1. Song song hoá có kiểm soát (worker pool đa luồng) + nhiều API key luân phiên để tăng thông lượng mà không vượt rate-limit từng key.
2. Pre-filter rẻ bằng NER nhẹ (spaCy) trước khi tốn lượt gọi LLM đắt tiền cho NER+RE đầy đủ.
3. Entity Resolution đổi FAISS `IndexFlatIP` (O(N²)) sang HNSW kèm blocking theo ký tự đầu/loại thực thể.
4. Neo4j: dùng `apoc.periodic.iterate` cho ghi lớn, tăng heap size, cân nhắc self-hosted thay vì Aura Free.
5. Ở quy mô này, **Super-node cap (degree > 100)** mới thực sự phát huy tác dụng — ở quy mô lab, độ phổ biến node cao nhất còn thấp hơn ngưỡng 100 nên chưa kiểm chứng được cơ chế cắt tỉa trong điều kiện thật.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | 80 batch trên 400 chunk: 202 chunk bị thay đổi, 77 chunk có `unresolved_mentions` (từ chối suy diễn khi mơ hồ — đúng thiết kế), 0 batch lỗi. 1 ca lỗi thật: model đôi lúc diễn giải lại câu thay vì chỉ thay thế đại từ (xem Phần 1, mục 1). |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Guard lọc cứng ngay sau khi nhận JSON từ LLM. Kết quả thật: **121 triple, 0 lỗi batch**, trích từ 100 chunk qua OpenRouter/`gpt-4o-mini`. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows AS row` theo batch 1000. Mỗi edge bắt buộc `source_chunk_id`, `published_date`, `evidence`, `confidence` — `graph_checks()` xác nhận `invalid_provenance_edges == 0`. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | 2 tầng: FAISS ANN (`threshold=0.85`) → Lexical Guard (`SequenceMatcher.ratio() >= 0.72`) → Union-Find. Thực tế 4 cặp vượt ngưỡng, cả 4 `MERGE_VECTOR` — phát hiện 1 ca đáng ngờ (`Chandrayaan-I` vs `-III`) là false merge tiềm ẩn (xem Phần 1, mục 2). |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50`, `GLOBAL_EDGE_CAP=250`, `MAX_GRAPH_CONTEXT_CHARS=14000` | `graph_supernode_events=0` trên toàn bộ 25 câu benchmark — cơ chế chưa từng cần kích hoạt vì đồ thị còn nhỏ (121 triple). |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` | 3 tiêu chí (1–5) + rationale, golden dataset thật 25 câu. Kết quả: Flat 1.84/2.08/1.84 vs Graph 1.80/1.92/1.80 (comp/faith/multihop) — GraphRAG thắng rõ nhất ở nhóm cross-doc và ở ca cụ thể multi-hop qua 2 nguồn (xem Phần 1, mục 4). |

---

### 2. Quá trình Debugging & Bài học

**Lỗi kỹ thuật phức tạp nhất:** không phải một lỗi đơn lẻ mà là **một chuỗi 5 lỗi dây chuyền** khi chuyển từ "code khung chạy được trên giấy" sang "chạy thật trên Google Colab":

1. **Xung đột dependency nhiều lớp:** `%pip install` không ghim version kéo về `transformers` bản mới, xung đột với `torch` cài sẵn trên Colab → `AttributeError: module 'torch' has no attribute '_utils'`. Ghim version cứng sửa được lỗi này nhưng lộ ra lỗi tiếp theo.
2. **Package "ma" không xóa được:** `peft` đóng gói sẵn trong base image Colab khiến `pip uninstall` không dọn sạch — `is_peft_available()` vẫn trả `True` dù đã "gỡ". Phải `rm -rf` thẳng thư mục site-packages còn sót.
3. **Nhầm cấu hình Neo4j Aura:** instance Free mới sinh `username`/`database` trùng Instance ID thay vì mặc định `neo4j` như tài liệu cũ — `AuthError` dù password đúng 100%.
4. **Dataset thực tế khác giả định code khung:** không có cột `text`/`content`/`article`, chỉ có `description` (~200 ký tự).
5. **Model bị Groq ngừng hỗ trợ, lỗi bị "ngụy trang":** `GROQ_MODEL=llama-3.3-70b-versatile` bị gỡ khỏi danh mục Groq → mọi request lỗi `404`, nhưng cơ chế retry+backoff (6 lần) khiến hiện tượng bề ngoài giống hệt "bị rate-limit" (~100s/batch) chứ không crash rõ ràng.

**Cách xử lý thành công:**
- Cô lập biến số bằng cell chẩn đoán tối giản (`max_retries=1`) để lỗi thật lộ ra ngay thay vì bị chôn trong retry.
- Gọi trực tiếp `groq_client.models.list()` để lấy danh sách model thật đang khả dụng thay vì tin tài liệu.
- Kiểm tra `raw_df.columns.tolist()` thực tế thay vì tin giả định trong code khung.
- Dùng `git fetch upstream` để lấy bộ golden dataset thật từ repo giảng viên, thay đổi cả chiến lược sampling (bỏ random-sample, lấy đúng 5000 dòng đầu, ưu tiên 28 dòng evidence vào ngân sách trích xuất).

**Bài học rút ra:**
1. Không tin tưởng mù quáng vào ví dụ/mặc định trong tài liệu — dịch vụ managed thay đổi hành vi mặc định theo thời gian.
2. Cơ chế retry/backoff có thể che giấu lỗi thật thành hiện tượng "chạy chậm" — luôn cần đường thoát chẩn đoán nhanh.
3. Không ghim version = rủi ro cực cao trên môi trường đã có sẵn nhiều package.
4. Luôn xác minh schema dữ liệu thật trước khi tin giả định trong code khung.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án:** AI Agent quản lý kênh cộng đồng đa nền tảng — bot cho Telegram và Discord, admin quản lý luật/quy định qua giao diện web (thêm luật thủ công hoặc upload nguyên file luật của kênh), agent tự động phát hiện và xử lý vi phạm (spam, ngôn từ thô tục, ...).

- **Đặc thù bài toán & Lý do chọn giải pháp:** Đồ án có 2 bài toán con đặc thù dữ liệu khác nhau → **Hybrid**, không chọn thuần một trong hai.
  1. *"Tin nhắn này vi phạm luật nào?"* — tra cứu đơn lẻ, luật là văn bản tự chứa → **Flat RAG** (vector search trên đoạn luật đã chunk) là đủ, dùng GraphRAG ở đây là over-engineering.
  2. *"User này vi phạm bao nhiêu lần, có nên leo thang hình phạt không, tài khoản Telegram/Discord này có phải cùng 1 người từng bị cảnh cáo ở kênh khác?"* — quan hệ đa bước thật sự giữa User → ViolationEvent → Rule → Punishment, có thể xuyên nhiều nền tảng → cần **GraphRAG**.

- **Cấu trúc Node & Relation dự kiến (nhánh lịch sử vi phạm):**
  - Nodes: `User`, `Channel`/`Server` (Telegram/Discord), `Rule`, `ViolationEvent`, `Punishment`
  - Relations: `(User)-[:POSTED]->(ViolationEvent)`, `(ViolationEvent)-[:VIOLATES]->(Rule)`, `(ViolationEvent)-[:OCCURRED_IN]->(Channel)`, `(Punishment)-[:RESULTED_FROM]->(ViolationEvent)`, `(User)-[:RECEIVED]->(Punishment)`, `(User)-[:MEMBER_OF]->(Channel)`
  - Khác với lab (nodes/edges do LLM extract từ văn bản tự do), ở đây phần lớn được hệ thống **ghi trực tiếp** từ sự kiện thật, giảm rủi ro hallucination lúc extraction so với ca `G5000-26` trong lab.

- **Chiến lược xử lý Super-node & Entity Resolution:**
  - *Entity Resolution* cho `User` giữa Telegram ↔ Discord rủi ro cao hơn lab: false merge = ban nhầm người vô tội (gán nhầm lịch sử vi phạm), false split = người vi phạm lách luật bằng tài khoản khác. Vì vậy **không tự động hoá bằng similarity** như lab (`MERGE_VECTOR` khi >0.85) — cần bằng chứng mạnh hơn (admin xác nhận thủ công hoặc OTP liên kết tài khoản). Bài học trực tiếp từ ca `Chandrayaan-I` vs `-III`: similarity cao không đồng nghĩa cùng thực thể.
  - *Super-node cap* áp dụng cho moderator lâu năm hoặc `Rule` phổ biến (VD "cấm spam") có hàng nghìn `ViolationEvent`. Khi tóm tắt lịch sử để quyết định hình phạt, cần cap số event đưa vào context (giống `SUPER_NODE_EDGE_CAP`) — nhưng khác tin tức, ranking không nên chỉ dựa `published_date`: vi phạm nghiêm trọng dù cũ vẫn cần giữ lại trong context.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | *(tự chấm)* | |
| Khả năng kiểm soát AI Coding Agent | *(tự chấm)* | 3 quyết định thật đã ghi lại ở Phần 1 mục 5 (2 từ chối, 1 đảo ngược có căn cứ khi có bằng chứng lỗi mới) |
| Chất lượng đồ thị tri thức xây dựng | *(tự chấm)* | 121 triple, 0 lỗi provenance, 0 false merge rõ ràng (trừ 1 ca nghi vấn Chandrayaan) |
| Khả năng phân tích và debug hệ thống | 5 | Xử lý thành công chuỗi 5 lỗi dây chuyền độc lập (dependency, package ma, auth, schema dataset, model deprecated) bằng phương pháp cô lập biến số và chẩn đoán trực tiếp thay vì đoán mò. |
