# Phân Tích Ca Lỗi (Failure Analysis) — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Thái Tú
**MSSV:** 2A202601504
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Nguồn dữ liệu:** `outputs/graphrag_eval_results.csv` (25 câu hỏi thật, golden dataset từ repo giảng viên)

---

## Ca lỗi 1 — GraphRAG hallucination do sai seed-entity matching

**Câu hỏi:** `G5000-26` — *"What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"*
**Reference answer:** Cohere (technology provider); chương trình customer-service agent hội thoại, và một hệ thống y tế tạo ghi chú lâm sàng.
**Điểm Judge:** Flat RAG 1/1/1, GraphRAG 1/1/1 (cả hai đều bị chấm thấp nhất, nhưng khác bản chất).

### Hiện tượng

- **Flat RAG** trả lời: *"not specified in the provided context"* — từ chối trả lời an toàn, không bịa thông tin.
- **GraphRAG** trả lời: khẳng định nhà cung cấp là **"Synopsys"** — một công ty chip design/EDA hoàn toàn không liên quan đến Amazon hay Cohere. Đây là hallucination thật: mô hình tự tin đưa ra một sự thật sai, không phải chỉ thiếu thông tin.

### Truy vết nguyên nhân gốc (root-cause)

1. **Loại trừ giả thuyết "super-node cap cắt nhầm cạnh":** cột `graph_supernode_events` của câu này bằng **0** — cơ chế cắt tỉa (`SUPER_NODE_DEGREE=100`) chưa từng kích hoạt ở quy mô đồ thị 121 triple của lab. Vậy lỗi không nằm ở tầng truy xuất/cắt tỉa cạnh.
2. **Bằng chứng thay vào đó trỏ về bước seed-entity matching** (`match_seeds()`, cell `3.2`): cùng một chunk Synopsys (`chunk_id=f6d852f34f2a34a17240::c0000`) xuất hiện lặp lại trong câu trả lời GraphRAG ở **3 câu hỏi hoàn toàn không liên quan tới nhau**:
   - `G5000-26` (về Amazon)
   - `G5000-30` (về Meta)
   - `G5000-49` (về Samsung)

   Đây không phải trùng hợp ngẫu nhiên một lần — việc cùng 1 chunk lặp lại ở 3 seed-entity khác nhau cho thấy **embedding của chunk Synopsys nằm gần nhiều câu hỏi khác nhau trong không gian vector**, nhiều khả năng vì nội dung chunk dùng dày đặc các từ khoá "AI"/"technology" chung chung, khiến bước match seed dựa trên cosine similarity chọn nhầm nó làm entry point traversal bất kể entity thật trong câu hỏi là gì.
3. Một khi seed sai được chọn, toàn bộ graph traversal xuất phát từ node sai → context trả về hoàn toàn lạc đề → LLM sinh câu trả lời tự tin nhưng sai (hallucination), thay vì việc thiếu context của Flat RAG khiến nó trả lời "không rõ".

### Tại sao đây là lỗi nghiêm trọng hơn lỗi của Flat RAG

Cả hai bị chấm cùng điểm 1/5, nhưng về mặt rủi ro sản phẩm, **hallucination tự tin nguy hiểm hơn từ chối trả lời** — người dùng cuối có thể tin nhầm "Synopsys" là câu trả lời đúng, trong khi câu trả lời "không đủ bằng chứng" của Flat RAG ít nhất trung thực về giới hạn của nó.

### Đề xuất khắc phục

1. Trong `match_seeds()`, ưu tiên exact/fuzzy string match tên entity trước, chỉ fallback sang vector similarity khi không tìm được match trực tiếp.
2. Thêm ngưỡng phân tách (margin) giữa candidate seed tốt nhất và candidate thứ nhì — nếu quá gần nhau (ambiguous), trả về `NO_SEED`/"không đủ bằng chứng" thay vì chọn liều, mô phỏng lại hành vi (vô tình) an toàn hơn của Flat RAG trong ca này.
3. Giám sát định kỳ các chunk bị match lặp lại bất thường trên nhiều câu hỏi không liên quan (như Synopsys ở đây) — đây là dấu hiệu sớm của một "embedding attractor" cần loại trừ hoặc re-embed.

---

## Ca lỗi 2 — Cả hai phương pháp cùng thất bại do giới hạn phạm vi trích xuất (extraction coverage)

**Câu hỏi:** `G5000-37` — *"What evidence shows Dell NativeEdge moving from a May product announcement to a July use-case description?"*
**Reference answer:** Bài Dell Technologies World tháng 5 nói về việc ra mắt NativeEdge; bài networking tháng 7 mô tả NativeEdge như một nền tảng triển khai ứng dụng edge.
**Điểm Judge:** Flat RAG 1/1/1, GraphRAG 1/1/1 — **cả hai cùng thất bại giống hệt nhau.**

### Hiện tượng

Cả hai phương pháp đều trả lời gần như nguyên văn giống nhau: *"The provided context does not contain any evidence regarding Dell NativeEdge... There are no specific references to Dell NativeEdge... in the supplied chunks."* Không có hallucination — cả hai đều thành thật thừa nhận thiếu bằng chứng.

### Truy vết nguyên nhân gốc

Đây **không phải** lỗi thuật toán retrieval (vector search sai, hay graph traversal sai) — vì cả 2 kiến trúc hoàn toàn độc lập về cơ chế truy xuất lại đưa ra cùng một kết luận "không có evidence". Nguyên nhân nhiều khả năng nằm ở **tầng dữ liệu đầu vào, phía trước cả 2 phương pháp**:

- Notebook giới hạn ngân sách trích xuất `EXTRACTION_MAX_CHUNKS=400` (trên tổng `GOLDEN_SCOPE_ROWS=5000` dòng), ưu tiên 28 dòng evidence đã biết trước vào ngân sách đó (`build_extraction_source()`).
- Tuy nhiên Flat RAG's FAISS index cũng được xây trên **cùng tập chunk đã chuẩn hoá** trong phạm vi 5000 dòng — nếu chunk chứa nội dung Dell NativeEdge tháng 5/tháng 7 nằm ngoài 400 chunk được ưu tiên xử lý sâu, hoặc bị `standardize_news`/`build_chunks` lọc bỏ do không đạt ngưỡng độ dài văn bản tối thiểu, thì **không phương pháp nào có cơ hội thấy được evidence này** — đây là giới hạn ở khâu chuẩn bị dữ liệu, không phải ở khâu suy luận/truy xuất.

### Đề xuất khắc phục

1. Xác minh trực tiếp: kiểm tra `evidence_row_ids_0based` của câu `G5000-37` trong golden dataset chi tiết, đối chiếu xem các `raw_row_id` đó có thực sự nằm trong `chunks_df` (sau bước `standardize_news`/`build_chunks`) hay đã bị lọc bỏ ở bước tiền xử lý.
2. Nếu bị lọc do thiếu cột text hợp lệ (cột `description` quá ngắn), cần nới điều kiện lọc hoặc bổ sung fallback ghép nhiều trường (title + description) để tăng tỉ lệ giữ được chunk chứa evidence.
3. Ở quy mô production, đây là lý do phải tách biệt rõ "coverage evaluation" (bao nhiêu % evidence rows của golden set thực sự lọt vào tập chunk đã xử lý) khỏi "retrieval accuracy evaluation" (trong số evidence có sẵn, phương pháp nào tìm đúng) — nếu không, một lỗi coverage ở tầng dữ liệu dễ bị hiểu nhầm thành lỗi thuật toán retrieval.

---

## Tổng kết

Hai ca lỗi minh hoạ **hai lớp nguyên nhân khác nhau** cần được xử lý bằng giải pháp khác nhau:

| Ca lỗi | Lớp nguyên nhân | Ảnh hưởng | Giải pháp |
|---|---|---|---|
| 1 (`G5000-26`) | Sai ở tầng thuật toán — seed-entity matching chọn nhầm entry point traversal | GraphRAG hallucination tự tin, nguy hiểm hơn "không biết" | Ưu tiên exact-match tên entity, thêm ngưỡng phân tách ambiguous |
| 2 (`G5000-37`) | Sai ở tầng dữ liệu — evidence chưa lọt vào tập chunk đã xử lý | Cả 2 phương pháp cùng thất bại như nhau | Tăng coverage tiền xử lý, tách riêng đo coverage vs. đo retrieval accuracy |
