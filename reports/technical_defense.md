# Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thái Tú
**MSSV:** 2A202601504
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19

---

## 1. Coreference Resolution

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

**Ca lỗi phụ (cùng họ vấn đề):** Chunk `fa44e0ddb529ef9fc5c9::c0000` — entity đúng (`Dar Al-Handasah Consultants – Dar`) nhưng thay thế máy móc, lặp lại nguyên cụm tên đầy đủ + tên viết tắt 3 lần trong 2 câu (`"Dar Al-Handasah Consultants – Dar's new Poland office"`), tạo văn bản ngữ pháp gượng — không sai entity nhưng thiếu bước làm sạch hậu kỳ.

**Đối chiếu — cơ chế hoạt động đúng:**
- Chunk `d38fc817b7dc3eeeb535::c0000` (trang chặn bot/captcha, không phải tin thật) → `unresolved_mentions: ['this', 'you', 'it']`. Model đúng đắn từ chối suy diễn khi không có antecedent rõ ràng.
- Chunk `69f2aecea98a9294feb6::c0000`: `"He issued this directive"` → `"Governor Mohammed Umar Bago issued this directive"` — thay thế sạch, chính xác, không mơ hồ.

Trên tổng 400 chunk gửi trích xuất: **202 chunk bị thay đổi**, **77 chunk có `unresolved_mentions`** (từ chối suy diễn), **0 batch lỗi hệ thống**.

---

## 2. Entity Resolution Threshold & Lexical Guard

**Ngưỡng cosine similarity:** `threshold = 0.85` (hạ từ mặc định `0.90` — lý do: mở rộng recall của candidate generation bằng vector ANN, đẩy toàn bộ gánh nặng chống false-merge sang Lexical Guard tầng sau — kiến trúc two-stage precision/recall).

**Kết quả thực tế** (`entity_resolution_audit.csv`, quy mô 400 chunk / 121 triple): chỉ có **4 cặp** vượt ngưỡng candidate generation, và **cả 4 đều được `MERGE_VECTOR`** — không có cặp nào bị `REJECT_GUARD` ở quy mô lab này (đồ thị còn nhỏ nên số cặp tên gần trùng cũng ít):

| Loại | Thực thể A | Thực thể B | Similarity | Quyết định |
|---|---|---|---|---|
| Company | Reliance Industries Ltd | Reliance Industries | 0.945 | MERGE_VECTOR |
| Company | Activision Blizzard | Activision Blizzard Inc. | 0.918 | MERGE_VECTOR |
| Company | Airbnb | Airbnb Inc. | 0.875 | MERGE_VECTOR |
| Technology | Chandrayaan-I | Chandrayaan-III | 0.858 | MERGE_VECTOR |

**Phát hiện đáng chú ý hơn một ca REJECT_GUARD "sạch":** cặp cuối cùng — `Chandrayaan-I` vs `Chandrayaan-III` (similarity 0.858) — theo tôi đây thực chất là một **false merge tiềm ẩn**, chứ không phải guard hoạt động đúng. Đây là 2 sứ mệnh không gian **khác nhau** của Ấn Độ (tàu đổ bộ Mặt Trăng ở các năm khác nhau), không phải cùng một thực thể viết khác dạng. Lexical Guard trong lab dùng `strip_suffix` (bỏ hậu tố doanh nghiệp như Inc/Corp/Ltd) rồi so `SequenceMatcher.ratio() >= 0.72` — cơ chế này được thiết kế để bắt hậu tố *doanh nghiệp*, không phải hậu tố *số thứ tự*. Vì "Chandrayaan-I" và "Chandrayaan-III" chỉ khác nhau đúng 1 ký tự số La Mã ở cuối, `SequenceMatcher` cho ratio rất cao → Guard không chặn được, dẫn tới gộp nhầm 2 thực thể khác nhau thành 1 node duy nhất trong graph.

**Bài học:** Lexical Guard hiện tại xử lý tốt lớp lỗi "cùng thực thể, viết khác dạng" (hậu tố công ty, viết tắt) nhưng **chưa xử lý được lớp lỗi "khác thực thể, tên gần giống theo số thứ tự/phiên bản"** (version, số La Mã, số thứ tự sự kiện/thế hệ sản phẩm). Đề xuất cải tiến: thêm rule riêng — nếu 2 tên chỉ khác nhau ở token số/số La Mã ở cuối chuỗi, tự động hạ điểm similarity hoặc bắt buộc review thủ công, thay vì tin tưởng hoàn toàn vào `SequenceMatcher.ratio()`.

---

## 3. Đồ thị & Super-node Mitigation

**Bằng chứng từ benchmark thực tế:** cột `graph_supernode_events` trong `graphrag_eval_results.csv` bằng **0 trên toàn bộ 25/25 câu hỏi** — nghĩa là ở quy mô lab này (400 chunk trích xuất → 121 triple), **không entity nào chạm ngưỡng `SUPER_NODE_DEGREE=100`**, cơ chế cắt tỉa chưa từng phải kích hoạt. Khớp với dự đoán trong `plan.md`: với chỉ ~121 triple, bậc kết nối cao nhất còn thấp hơn nhiều so với ngưỡng 100.

> **TODO nhỏ còn thiếu:** số liệu top-3 degree thật (`top_degree_df` cell `2.4` / `test_supernode_policy()` cell `5.1`) — gửi output cell chẩn đoán đã đưa trước đó để điền nốt bảng dưới.

| Hạng | Tên thực thể | Loại | Degree |
|---|---|---|---|
| 1 | *(chờ dữ liệu)* | | |
| 2 | | | |
| 3 | | | |

**Ưu điểm của việc lấy 50 cạnh mới nhất theo `published_date DESC`:** giới hạn context bounded (tránh nổ token khi 1 entity có hàng trăm cạnh), ưu tiên thông tin cập nhật — hợp với tin tức công nghệ vốn thay đổi nhanh (VD: trạng thái "đang cân nhắc" của AWS với chip AMD có thể đã trở thành quyết định cuối cùng ở bài báo mới hơn).

**Rủi ro quan sát được (gián tiếp, qua ca lỗi G5000-26 — xem mục 4):** vấn đề thực tế gặp trong benchmark **không phải** do cơ chế cắt tỉa xoá mất cạnh đúng (`graph_supernode_events=0` chứng minh cơ chế này chưa từng kích hoạt), mà nằm ở **bước match seed-entity phía trước**: câu hỏi về "Amazon" bị traversal lệch sang một chunk hoàn toàn không liên quan (Synopsys/chip design). Tức là ở quy mô lab, rủi ro thực tế lớn hơn super-node degree cap lại là **độ chính xác của seed matching**. Nếu scale lên production, cả 2 lớp rủi ro (seed matching sai + super-node cap cắt nhầm cạnh lịch sử quan trọng) đều cần giám sát song song, không chỉ riêng super-node như giả định ban đầu.

**Lưu ý kỹ thuật:** `coalesce(r.published_date,'')` trong `recent_edges()` đẩy các cạnh thiếu ngày xuống cuối danh sách sắp xếp → luôn bị cắt trước tiên khi vượt cap, có thể làm mất đúng những cạnh cần thiết nếu extraction bị lỗi không gán được ngày.

---

## 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Golden dataset dùng cho benchmark: **25 câu hỏi thật** (2 factoid, 12 multi-hop, 11 cross-doc), lấy từ repo gốc giảng viên (`VinUni-AI20k/K4-Track3-Lab19-GraphRAG`), neo vào 5000 dòng đầu tiên của `hackernoon_subset.csv` với `reference_answer` và `evidence_row_ids_0based` đã có sẵn — không phải tự bịa câu hỏi, đảm bảo đánh giá có căn cứ đối chiếu khách quan.

### Bảng benchmark (LLM-as-a-Judge, thang 1–5), theo nhóm câu hỏi

| Nhóm | Comprehensiveness | Faithfulness | Multi-hop reasoning | Latency (s) | Token usage |
|---|---|---|---|---|---|
| factoid (n=2) | Flat **2.00** / Graph **1.50** | Flat **2.00** / Graph **1.50** | Flat **2.00** / Graph **1.50** | Flat 1.25 / Graph 0.87 | Flat 572.5 / Graph 452.5 |
| multi-hop (n=12) | Flat **1.92** / Graph **1.83** | Flat **2.25** / Graph **1.92** | Flat **1.92** / Graph **1.83** | Flat 2.39 / Graph 2.49 | Flat 682.5 / Graph 665.3 |
| cross-doc (n=11) | Flat **1.73** / Graph **1.82** | Flat **1.91** / Graph **2.00** | Flat **1.73** / Graph **1.82** | Flat 1.62 / Graph 1.64 | Flat 600.5 / Graph 472.7 |
| **Tổng (n=25, weighted)** | Flat **1.84** / Graph **1.80** | Flat **2.08** / Graph **1.92** | Flat **1.84** / Graph **1.80** | Flat 1.96 / Graph 1.99 | Flat 637.6 / Graph 563.6 |

**Nhận xét trung thực:** ở quy mô lab (400 chunk / 121 triple / 25 câu hỏi), GraphRAG **không** cho thấy lợi thế điểm số đồng đều so với Flat RAG — thậm chí thấp hơn nhẹ ở comprehensiveness/faithfulness tổng thể, ngoại trừ nhóm `cross-doc` (nơi GraphRAG nhỉnh hơn, đúng như kỳ vọng lý thuyết vì đây là nhóm cần nối thông tin từ nhiều tài liệu). Điểm mạnh thực sự của GraphRAG lộ rõ ở **ca cụ thể** (xem case 1 dưới) chứ không phải cải thiện trung bình trên mọi loại câu hỏi — nguyên nhân chính: ngân sách trích xuất giới hạn (400/5000 chunk, ưu tiên 28 dòng evidence) khiến nhiều bằng chứng liên quan **chưa kịp vào graph dưới dạng triple**, trong khi Flat RAG index được xây trên toàn bộ chunk nên vẫn "chạm" được nhiều đoạn văn thô hơn về mặt retrieval — dù không tổng hợp đa bước được. Điểm tuyệt đối còn thấp (1.5–2.1/5) ở cả hai phương pháp phản ánh đúng giới hạn quy mô lab, không phải lỗi kiến trúc.

### Ca 1 — GraphRAG thành công rõ rệt, Flat RAG thất bại: `G5000-44`

**Câu hỏi:** *"What two distinct partner ecosystems connect L&T Technology Services to advanced infrastructure in 2023: one for urban-rail 5G and one for OT security?"*
**Reference:** Thales + Qualcomm (urban-rail 5G) **và** Palo Alto Networks, vai trò MSSP (OT security) — 2 sự kiện tách biệt, nằm ở 2 dòng dữ liệu gốc khác nhau (evidence rows 261 và 891 trong golden dataset).

| | Comprehensiveness | Faithfulness | Multi-hop |
|---|---|---|---|
| Flat RAG | 2 | 2 | 2 |
| GraphRAG | **4** | **5** | **4** |

- **Flat RAG thất bại vì:** chỉ retrieve được đoạn nói về Thales (5G), tự nhận *"specific details about the OT security partner are not provided in the context"* — vector search single-hop chỉ khớp được 1 trong 2 chunk liên quan, không có cơ chế nào buộc nó tìm tiếp chunk thứ 2 về cùng công ty.
- **GraphRAG thành công vì:** trả lời đúng **cả 2** — Thales cho 5G *và* Palo Alto Networks cho OT security — vì graph traversal đi từ node "L&T Technology Services" nối ra 2 cạnh riêng biệt (ứng với 2 bài báo khác nguồn), đúng cơ chế multi-hop mà kiến trúc GraphRAG được thiết kế cho: gộp nhiều cạnh cùng xuất phát từ 1 entity thay vì chỉ dựa vào độ tương đồng ngữ nghĩa của riêng câu hỏi.

### Ca 2 — GraphRAG thất bại (hallucination do sai seed-matching): `G5000-26`

**Câu hỏi:** *"What external technology provider is named inside Amazon's July AI-service expansion...?"*
**Reference:** Cohere.

| | Comprehensiveness | Faithfulness | Multi-hop |
|---|---|---|---|
| Flat RAG | 1 | 1 | 1 |
| GraphRAG | 1 | 1 | 1 |

Cả hai đều bị chấm 1/5 nhưng **theo hai cách khác nhau về bản chất**:
- **Flat RAG:** trả lời *"not specified in the provided context"* — an toàn, không bịa, chỉ là recall thấp (không retrieve đúng chunk).
- **GraphRAG:** trả lời **sai và tự tin** — khẳng định nhà cung cấp là *"Synopsys"* (một công ty chip design/EDA hoàn toàn không liên quan tới câu hỏi về Amazon/Cohere). Đây là hallucination thật, tệ hơn một câu từ chối trả lời.

**Root cause (đã xác minh qua dữ liệu, không suy đoán):** cùng một chunk Synopsys (`chunk_id=f6d852f34f2a34a17240::c0000`) xuất hiện lặp lại một cách bất thường trong câu trả lời GraphRAG ở **ít nhất 3 câu hỏi không liên quan tới nhau** (`G5000-26` về Amazon, `G5000-30` về Meta, `G5000-49` về Samsung) — trong khi `graph_supernode_events=0` ở cả 3 câu, tức **không phải** do super-node degree cap cắt nhầm cạnh (xem mục 3). Nguyên nhân thực sự là **seed-entity matching** (`match_seeds()`, cell `3.2`) khớp nhầm: chunk Synopsys có nội dung dùng nhiều từ khoá "AI" chung chung, khiến vector embedding của nó vô tình gần với embedding của nhiều câu hỏi khác nhau về "AI service/AI capability", bất kể entity thật trong câu hỏi là Amazon hay Meta hay Samsung.

**Đề xuất khắc phục:** (1) tăng trọng số cho exact-match tên entity (không chỉ dựa vector similarity) khi `match_seeds()` chọn seed ban đầu; (2) thêm ngưỡng tối thiểu giữa similarity của candidate tốt nhất và candidate thứ nhì — nếu quá gần nhau (ambiguous), nên trả lời "không đủ bằng chứng" thay vì chọn liều một seed sai, giống cách Flat RAG (vô tình) làm đúng hơn trong ca này.

---

## 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

### Bảng đánh đổi định lượng (từ `graphrag_eval_results.csv`, 25 câu hỏi thật)

| Metric (trung bình) | Flat RAG | GraphRAG | Δ | Nhận xét |
|---|---|---|---|---|
| Comprehensiveness | 1.84 | 1.80 | -0.04 | Gần như ngang nhau ở quy mô lab |
| Faithfulness | 2.08 | 1.92 | -0.16 | Flat nhỉnh hơn một chút (xem lý do coverage ở mục 4) |
| Multi-hop reasoning | 1.84 | 1.80 | -0.04 | Gần như ngang nhau |
| Latency (s) | 1.96 | 1.99 | +0.03 (~1.5%) | GraphRAG chậm hơn không đáng kể dù có thêm bước seed-match + Cypher query |
| Token usage | 637.6 | 563.6 | -74.0 (**-11.6%**) | GraphRAG **rẻ hơn** — context dạng triple cô đọng hơn đoạn văn thô của Flat RAG |

**Kết luận đánh đổi:** ở quy mô lab, chi phí indexing/vận hành của GraphRAG (Neo4j, entity resolution, seed matching) **không đổi lấy được** một khoảng cách điểm số rõ ràng — điểm mạnh của GraphRAG chỉ lộ ra ở đúng dạng câu hỏi nó được thiết kế cho (nối 2 sự kiện rời rạc về cùng 1 entity, xem case `G5000-44` mục 4), không phải cải thiện đều trên mọi câu hỏi. Điều này cho thấy đánh đổi GraphRAG chỉ thực sự "lời" khi tập dữ liệu đủ lớn để nhiều câu hỏi rơi đúng vào nhóm cross-doc/multi-hop cần nối nhiều nguồn — ở tập nhỏ, overhead indexing chưa được khấu hao.

**Quyết định của AI Coding Agent (3 ví dụ thật, không giả định):**

1. Khi tốc độ pipeline bị chậm bất thường (~100 giây/batch ở bước coreference), AI Agent đề xuất chuyển hẳn sang gọi song song (`ThreadPoolExecutor`) để tăng tốc ngay từ sớm trong buổi — **tôi từ chối** vì lúc đó chưa xác định được nguyên nhân gốc (hoá ra là model đã bị Groq ngừng hỗ trợ, không phải do gọi tuần tự), và không muốn thêm một biến số thay đổi (concurrency) trong lúc đang debug một vấn đề khác. Về sau, khi thực sự thử concurrency ở bước extraction, dữ liệu thực tế (thông lượng thấp hơn cả chạy tuần tự do rate-limit contention) xác nhận quyết định trì hoãn ban đầu là đúng — vấn đề gốc không nằm ở tốc độ gọi mà ở model bị deprecate.
2. AI Agent ban đầu đề xuất chuyển bước LLM-as-a-Judge sang OpenAI/OpenRouter — **tôi từ chối lúc đầu** vì muốn giữ kiến trúc đơn giản (1 provider duy nhất là Groq). Quyết định này **sau đó bị đảo ngược có căn cứ**: Judge model trên Groq (`qwen/qwen3.6-27b`) thực sự lỗi thật giữa lần chạy đánh giá 25 câu (`json_validate_failed` — hết ngân sách completion token trước khi sinh xong JSON hợp lệ). Khi bằng chứng lỗi cụ thể xuất hiện (không phải phỏng đoán), tôi đồng ý đổi Judge sang OpenRouter — bài học: từ chối một đề xuất "cho gọn kiến trúc" là đúng khi chưa có bằng chứng, nhưng phải sẵn sàng đảo ngược khi có lỗi thật, không giữ quyết định cũ vì cố chấp.
3. Khi `.gitignore` có nguy cơ chặn nhầm 2 file CSV bắt buộc nộp bài, Agent đề xuất bỏ hẳn rule chặn `*.csv` cho gọn — tôi chọn phương án thêm rule allow-list cụ thể (`!outputs/*.csv`) thay vì xoá rule chặn gốc, để tránh vô tình commit nhầm file dataset 300MB (`hackernoon_subset.csv`) lên GitHub.

**Giải pháp scale lên 350MB (~100,000 bài báo):**

Bottleneck đầu tiên **không phải Neo4j** mà là **thông lượng LLM extraction**. Ước tính: 100k bài × vài chunk/bài ≈ vài trăm nghìn chunk; ở batch size 4/request và rate-limit free-tier (~30 RPM), chạy tuần tự đơn luồng sẽ mất hàng chục ngày — hoàn toàn không khả thi.

Giải pháp kiến trúc:
1. **Song song hoá có kiểm soát** (worker pool đa luồng, đã áp dụng ở quy mô nhỏ trong lab này cho bước extraction/evaluation) kết hợp nhiều API key luân phiên để tăng thông lượng mà không vượt rate-limit từng key.
2. **Pre-filter rẻ trước khi gửi LLM:** dùng NER nhẹ (spaCy) lọc chunk có ít nhất 2 thực thể tiềm năng trước khi tốn lượt gọi LLM đắt tiền cho NER+RE đầy đủ.
3. **Entity Resolution đổi FAISS `IndexFlatIP` (O(N²) khi search) sang HNSW** kèm blocking theo ký tự đầu/loại thực thể, vì ở quy mô hàng trăm nghìn mention, Flat index sẽ không còn khả thi về thời gian.
4. **Neo4j:** dùng `apoc.periodic.iterate` cho các thao tác ghi lớn, tăng heap size, cân nhắc self-hosted thay vì Aura Free (giới hạn tài nguyên).
5. Ở quy mô này, chính sách **Super-node cap (degree > 100)** mới thực sự phát huy tác dụng — ở quy mô lab (400 chunk), độ phổ biến node cao nhất còn thấp hơn ngưỡng 100, nên cơ chế cắt tỉa cần được kiểm chứng bằng cách hạ tạm ngưỡng thử nghiệm (xem mục 3).
