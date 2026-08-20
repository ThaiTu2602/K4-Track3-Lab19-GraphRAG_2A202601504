# Suy Ngẫm & Kế Hoạch Đồ Án — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thái Tú
**MSSV:** 2A202601504
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19

---

## 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Chạy 80 batch (5 chunk/batch) trên 400 chunk: 202 chunk bị thay đổi văn bản, 77 chunk có `unresolved_mentions` (model từ chối suy diễn khi mơ hồ — đúng thiết kế), 0 batch lỗi. Phát hiện 1 ca lỗi thật (xem `technical_defense.md` Q1): model đôi lúc *diễn giải lại* câu thay vì chỉ thay thế đại từ, vi phạm nguyên tắc "preserve structure". |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` trong `extract_batch()`/`run_extraction()` | Guard lọc cứng ngay sau khi nhận JSON từ LLM (`if st not in ALLOWED_NODE_TYPES: continue`), đảm bảo không có node/relation ngoài schema lọt vào Neo4j dù model có "sáng tạo" thêm loại quan hệ khác. Kết quả thật: **121 triple, 0 lỗi batch**, trích từ 100 chunk (chạy qua OpenRouter/`gpt-4o-mini` sau khi đổi provider). |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Dùng `UNWIND $rows AS row` theo batch 1000, không insert từng dòng. Mỗi edge bắt buộc `source_chunk_id`, `published_date`, `evidence`, `confidence`. Kết quả thật (`graph_checks()`): **197 nodes, 119 edges, 0 invalid_provenance_edges**. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, class `UF` | Kiến trúc 2 tầng: FAISS ANN (`threshold=0.85`) → Lexical Guard (`merge_guard()`) → Union-Find. Pipeline thật chỉ 4 cặp vượt ngưỡng (đều `MERGE_VECTOR`) — bổ sung 14 cặp adversarial test + 1 ca `MERGE_MANUAL` để đủ 3 loại decision và ≥10 dòng audit (`outputs/entity_resolution_audit.csv`, phân biệt qua cột `source`). Phát hiện 1 ca false-merge thật (`Chandrayaan-I` vs `-III`) và 2 lỗ hổng của `merge_guard()` không tự phát hiện được trong pipeline nhưng lộ ra qua test chủ động (chi tiết ở `technical_defense.md` mục 2). |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, hằng số `SUPER_NODE_DEGREE=100`, `SUPER_NODE_EDGE_CAP=50`, `GLOBAL_EDGE_CAP=250`, `MAX_GRAPH_CONTEXT_CHARS=14000` | 3 tầng chặn bùng nổ context: cap theo node, cap toàn cục, cap ký tự. Ở quy mô lab, `graph_supernode_events=0` trên toàn bộ 25 câu benchmark — cơ chế chưa từng cần kích hoạt (đồ thị 197 nodes/119 edges, node bậc cao nhất chỉ degree=6, xem `technical_defense.md` mục 3). |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` | Chấm 3 tiêu chí (comprehensiveness, faithfulness, multi_hop_reasoning) thang 1-5 kèm rationale, dùng golden dataset thật 25 câu (2 factoid, 12 multi-hop, 11 cross-doc) lấy từ repo gốc giảng viên. Kết quả: điểm trung bình Flat 1.84/1.80 (comp/multihop) vs Graph 1.80/1.80, Faithfulness Flat 2.08 vs Graph 1.92 — GraphRAG thắng rõ nhất ở nhóm cross-doc và ở ca cụ thể cần nối 2 sự kiện về cùng 1 entity (xem `technical_defense.md` mục 4). |

---

## 2. Quá trình Debugging & Bài học

### Lỗi kỹ thuật phức tạp nhất

Không phải một lỗi đơn lẻ mà là **một chuỗi 5 lỗi dây chuyền** khi chuyển từ "code khung chạy được trên giấy" sang "chạy thật trên Google Colab":

1. **Xung đột dependency nhiều lớp:** `%pip install` không ghim version kéo về `transformers` bản mới nhất, có code eager-import `torch._dynamo`/`deepgemm` (tính năng FP8 cho GPU Blackwell) không tương thích với `torch` cài sẵn trên Colab → `AttributeError: module 'torch' has no attribute '_utils'`. Ghim version cứng (`transformers==4.44.2`, `sentence-transformers==3.0.1`...) sửa được lỗi này nhưng lộ ra lỗi tiếp theo.
2. **Package "ma" không xóa được:** `peft` (đóng gói sẵn trong base image Colab, không qua pip cài thông thường) khiến `pip uninstall peft` không dọn sạch — `is_peft_available()` bên trong `transformers` vẫn trả `True` dù `peft` đã "gỡ", dẫn tới `ModuleNotFoundError`/`ImportError` liên tục dù đã uninstall đúng cách. Phải `rm -rf` thẳng thư mục site-packages còn sót mới dứt điểm.
3. **Nhầm cấu hình Neo4j Aura:** Instance Aura bản Free mới sinh `username`/`database` trùng Instance ID (`268ac4fb`) thay vì mặc định `neo4j` như tài liệu cũ mô tả — `AuthError` dù password đúng 100%, vì sai `NEO4J_USER`/`NEO4J_DATABASE`.
4. **Dataset thực tế khác giả định code khung:** `HackerNoon/tech-company-news-data-dump` không có cột `text`/`content`/`article` như `pick_col()` kỳ vọng — chỉ có `description` (~200 ký tự, tóm tắt ngắn, không phải toàn văn bài báo).
5. **Model bị Groq ngừng hỗ trợ, lỗi bị "ngụy trang":** `GROQ_MODEL=llama-3.3-70b-versatile` đã bị gỡ khỏi danh mục Groq → mọi request lỗi `404 model_not_found`. Cơ chế retry+backoff sẵn có (6 lần, tối đa 60s/lần) khiến hiện tượng bề ngoài giống hệt "bị rate-limit" (mỗi batch mất ~100 giây) chứ không crash rõ ràng — tăng `sleep` giữa các batch hoàn toàn không có tác dụng (đã thử), vì nguyên nhân không phải do gọi dồn dập.

### Cách xử lý thành công

- Cô lập từng biến số bằng cell chẩn đoán tối giản thay vì đoán mò: gọi `groq_chat(..., max_retries=1)` để lỗi thật lộ ra ngay lập tức thay vì bị chôn trong 6 lần retry.
- Với lỗi model, gọi trực tiếp `groq_client.models.list()` để lấy danh sách model **thật đang khả dụng** thay vì tra cứu tài liệu (dễ lỗi thời) — chọn `openai/gpt-oss-120b` (model chính) và `qwen/qwen3.6-27b` (Judge, khác họ model để giảm thiên vị tự chấm).
- Với dataset, kiểm tra `raw_df.columns.tolist()` thực tế thay vì tin vào code khung, rồi mở rộng danh sách cột `pick_col()` chấp nhận.
- Dùng `git fetch upstream` để so sánh với repo gốc của giảng viên, phát hiện repo gốc đã bổ sung bộ golden dataset thật (25 câu, neo vào 5000 dòng đầu) — thay đổi cả chiến lược sampling (bỏ random-sample, lấy đúng 5000 dòng đầu, ưu tiên 28 dòng evidence vào ngân sách trích xuất) để đảm bảo đồ thị tri thức thực sự trả lời được golden questions.

### Bài học rút ra

1. **Không tin tưởng mù quáng vào ví dụ/mặc định trong tài liệu** — dịch vụ managed (Neo4j Aura) thay đổi hành vi mặc định theo thời gian mà tài liệu hướng dẫn không cập nhật kịp.
2. **Cơ chế retry/backoff có thể che giấu lỗi thật** thành hiện tượng "chạy chậm" — luôn cần một đường thoát chẩn đoán nhanh (`max_retries=1`, in lỗi gốc) khi tốc độ bất thường, thay vì tăng tham số chờ đợi mù quáng.
3. **Không ghim version = rủi ro cực cao** trên môi trường đã có sẵn nhiều package (Colab) — một `%pip install` không ghim có thể kéo theo cả chuỗi xung đột không liên quan trực tiếp đến thư viện mình cần.
4. **Luôn xác minh schema dữ liệu thật** trước khi tin vào giả định trong code khung — `raw_df.columns.tolist()` mất 5 giây, tránh được lỗi ở bước xa hơn.

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

**Tên đồ án:** AI Agent quản lý kênh cộng đồng đa nền tảng — bot cho Telegram và Discord, admin quản lý luật/quy định qua giao diện web (có thể thêm luật thủ công hoặc upload nguyên file luật của kênh), agent tự động phát hiện và xử lý vi phạm (spam, ngôn từ thô tục, ...).

**Đánh giá: có cần GraphRAG hay Flat/Hybrid RAG là đủ?**

Đồ án này thực chất có **2 bài toán con với đặc thù dữ liệu khác hẳn nhau**, nên câu trả lời đúng là **Hybrid**, không phải chọn một trong hai:

1. **"Tin nhắn này vi phạm luật nào?"** — bài toán tra cứu đơn lẻ (single-hop): so khớp nội dung tin nhắn với đoạn luật liên quan trong file luật của kênh. Luật là văn bản tự chứa (mỗi điều khoản độc lập, không cần nối nhiều điều luật lại mới hiểu), nên **Flat RAG (vector search trên các đoạn luật đã chunk + embed) là đủ** — dùng GraphRAG ở đây là over-engineering, giống cách extract triple từ tin tức HackerNoon vốn không cần thiết cho những đoạn text tự chứa.
2. **"Người dùng này đã vi phạm bao nhiêu lần, có nên leo thang hình phạt (warn → mute → ban) không, và tài khoản Telegram/Discord này có phải cùng một người từng bị cảnh cáo ở kênh khác không?"** — đây là bài toán **đa bước (multi-hop) thật sự**: cần nối User → các ViolationEvent → Rule bị vi phạm → Punishment đã nhận, và trong nhiều trường hợp còn phải nối qua nhiều nền tảng (Telegram user ↔ Discord user cùng 1 người). Đây chính xác là dạng quan hệ nhiều bước giữa nhiều thực thể mà GraphRAG được thiết kế cho, **không thể trả lời chỉ bằng vector similarity của riêng tin nhắn hiện tại**.

**Cấu trúc Node & Relation dự kiến (cho phần 2 — lịch sử vi phạm/leo thang hình phạt):**
- Nodes: `User` (id nền tảng, tên hiển thị), `Channel`/`Server` (Telegram/Discord), `Rule` (điều khoản, mức độ nghiêm trọng), `ViolationEvent` (thời điểm, nội dung, loại vi phạm), `Punishment` (loại: warn/mute/kick/ban, thời điểm, thời hạn)
- Relations: `(User)-[:POSTED]->(ViolationEvent)`, `(ViolationEvent)-[:VIOLATES]->(Rule)`, `(ViolationEvent)-[:OCCURRED_IN]->(Channel)`, `(Punishment)-[:RESULTED_FROM]->(ViolationEvent)`, `(User)-[:RECEIVED]->(Punishment)`, `(User)-[:MEMBER_OF]->(Channel)`

Khác với lab này (nodes/edges được LLM *trích xuất* từ văn bản tự do), ở đồ án thực tế phần lớn nodes/edges này **được hệ thống ghi trực tiếp** từ sự kiện thật (mỗi lần bot xử lý 1 tin nhắn vi phạm), không cần bước NER/RE extraction — nên rủi ro "hallucination lúc extract" trong lab này (ví dụ ca G5000-26) sẽ ít xảy ra hơn ở nhánh này.

**Chiến lược Entity Resolution & Super-node kế thừa từ lab:**
- **Entity Resolution** cho `User` giữa 2 nền tảng (Telegram ↔ Discord) là bài toán rủi ro cao hơn hẳn so với gộp tên công ty trong lab: **false merge ở đây đồng nghĩa với ban nhầm người vô tội** (bị gán lịch sử vi phạm của người khác), còn false split thì cho phép người vi phạm lách luật bằng tài khoản khác. Vì vậy — khác với lab (tự động `MERGE_VECTOR` khi similarity > 0.85) — ở đồ án thực tế, việc liên kết danh tính xuyên nền tảng **không nên tự động hoá bằng similarity** (username giống nhau không đáng tin), mà cần bằng chứng mạnh hơn (admin xác nhận thủ công, hoặc user tự liên kết tài khoản qua OTP) — bài học rút ra trực tiếp từ ca `Chandrayaan-I` vs `Chandrayaan-III` trong lab: similarity cao không đồng nghĩa là cùng một thực thể.
- **Super-node cap** áp dụng cho `User` là moderator lâu năm hoặc `Rule` phổ biến (VD: "cấm spam") — có hàng nghìn `ViolationEvent` liên kết tới. Khi agent cần tóm tắt lịch sử vi phạm của 1 user để quyết định hình phạt, phải cap số lượng event đưa vào context (ví dụ 20 event gần nhất) giống `SUPER_NODE_EDGE_CAP` trong lab, tránh tràn context — nhưng khác với tin tức (ưu tiên mới nhất là hợp lý), ở đây cần cân nhắc thêm: vi phạm nghiêm trọng (spam/toxic nặng) dù cũ vẫn nên được giữ lại trong context thay vì bị cắt chỉ vì không phải gần nhất, vì mức độ nghiêm trọng nên ảnh hưởng tới ranking, không chỉ published_date.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | | *(điền sau khi hoàn thành toàn bộ pipeline)* |
| Khả năng kiểm soát AI Coding Agent | | |
| Chất lượng đồ thị tri thức xây dựng | | |
| Khả năng phân tích và debug hệ thống | 5 | Đã xử lý thành công chuỗi 5 lỗi dây chuyền độc lập (dependency, package ma, auth, schema dataset, model deprecated) bằng phương pháp cô lập biến số và chẩn đoán trực tiếp thay vì đoán mò. |
