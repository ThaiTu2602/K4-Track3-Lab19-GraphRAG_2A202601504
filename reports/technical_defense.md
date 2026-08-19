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

> **TODO — điền sau khi chạy xong cell `2.2`:**
> - Ngưỡng cosine similarity đã dùng: `threshold = 0.85` (đã hạ từ mặc định `0.90` — lý do: mở rộng recall của bước candidate generation bằng vector ANN, đẩy toàn bộ gánh nặng chống false-merge sang Lexical Guard tầng sau — kiến trúc two-stage precision/recall).
> - Cần: 1 cặp thực thể có `similarity > 0.85` nhưng bị `REJECT_GUARD` — lấy từ `entity_resolution_audit_df.query("decision=='REJECT_GUARD'").sort_values("similarity", ascending=False).head()`.
> - Giải thích lý do bị chặn (ví dụ: hậu tố công ty không rút gọn được, hoặc tên người trùng họ).

---

## 3. Đồ thị & Super-node Mitigation

> **TODO — điền sau khi chạy xong cell `5.1` (`test_supernode_policy()`):**
> - Top 3 thực thể bậc cao nhất: tên, loại (Company/Person/Technology), degree — lấy từ `top_degree_df` (cell `2.4`).
> - Ưu điểm của việc lấy 50 cạnh mới nhất theo `published_date DESC`: giới hạn context bounded, ưu tiên thông tin cập nhật — hợp với tin tức công nghệ vốn thay đổi nhanh.
> - Rủi ro: câu hỏi về sự kiện lịch sử xa (VD: "Microsoft mua công ty nào năm 2016?") có thể bị cắt mất toàn bộ cạnh liên quan nếu node đó là super-node — silent failure, model sẽ trả lời "không đủ bằng chứng" dù graph thực sự có dữ liệu. Đề xuất khắc phục: query-aware selection (nếu câu hỏi chứa mốc thời gian cụ thể, đổi `ORDER BY` thành gần mốc thời gian đó thay vì luôn ưu tiên mới nhất).
> - Lưu ý kỹ thuật: `coalesce(r.published_date,'')` trong `recent_edges()` đẩy các cạnh thiếu ngày xuống cuối danh sách sắp xếp → luôn bị cắt trước tiên khi vượt cap, có thể làm mất đúng những cạnh cần thiết nếu extraction bị lỗi không gán được ngày.

---

## 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

> **TODO — điền sau khi chạy xong cell `4.4` (`comparison_table()` + export CSV):**
>
> Bảng benchmark 5 tiêu chí (Comprehensiveness, Faithfulness, Multi-hop Reasoning, Latency, Token usage) theo 3 nhóm câu hỏi (factoid/multi-hop/cross-doc), lấy từ `comparison_df`.
>
> **2 ca lỗi điển hình cần trích từ `eval_results_df`:**
> 1. Ca Flat RAG thất bại, GraphRAG thành công: chọn 1 câu nhóm `multi-hop` hoặc `cross-doc` có `graph_comprehensiveness` hoặc `graph_faithfulness` cao hơn rõ rệt (`delta >= 0.75`) so với `flat_*`. Chứng minh bằng cách in `flat["retrieved"].chunk_id` — nếu chunk chứa thông tin hop thứ 2 KHÔNG nằm trong top-k Flat RAG lấy được, đó chính là root cause (vector search không kết nối được 2 chunk rời rạc).
> 2. Ca GraphRAG thất bại/khó khăn: thường là factoid đơn giản (graph chỉ giữ triple đã lược ngữ cảnh, mất sắc thái so với đoạn văn gốc) hoặc trường hợp `NO_SEED` (câu hỏi không chứa tên thực thể rõ ràng để `match_seeds()` bắt được) — kiểm tra `graph_debug.diagnostics.reason`.

Golden dataset dùng cho benchmark: **25 câu hỏi thật** (2 factoid, 12 multi-hop, 11 cross-doc), lấy từ repo gốc giảng viên (`VinUni-AI20k/K4-Track3-Lab19-GraphRAG`), neo vào 5000 dòng đầu tiên của `hackernoon_subset.csv` với `reference_answer` và `evidence_row_ids_0based` đã có sẵn — không phải tự bịa câu hỏi, đảm bảo đánh giá có căn cứ đối chiếu khách quan.

---

## 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

> **TODO (một phần) — điền số liệu Latency/Token trung bình sau khi có `eval_results_df`.**

**Quyết định từ chối đề xuất của AI Coding Agent (2 ví dụ thật, không giả định):**

1. Khi tốc độ pipeline bị chậm bất thường (~100 giây/batch ở bước coreference), AI Agent đề xuất chuyển hẳn sang gọi song song (`ThreadPoolExecutor`) để tăng tốc ngay từ sớm trong buổi — **tôi từ chối** vì lúc đó chưa xác định được nguyên nhân gốc (hoá ra là model đã bị Groq ngừng hỗ trợ, không phải do gọi tuần tự), và không muốn thêm một biến số thay đổi (concurrency) trong lúc đang debug một vấn đề khác, tránh làm rối thêm việc chẩn đoán. Agent tôn trọng và không tự ý áp dụng.
2. AI Agent đề xuất đổi bước LLM-as-a-Judge sang dùng OpenAI (`gpt-4o-mini`) để tránh rate-limit Groq — **tôi từ chối** phần lớn vì muốn giữ kiến trúc đơn giản, chỉ 1 provider duy nhất; đồng thời việc đó chỉ ảnh hưởng ~16% tổng số lượt gọi LLM của pipeline (phần lớn coref/extraction/answer generation hardcode gọi Groq), không giải quyết được nút thắt cổ chai thật sự.
3. Khi `.gitignore` có nguy cơ chặn nhầm 2 file CSV bắt buộc nộp bài, Agent đề xuất bỏ hẳn rule chặn `*.csv` cho gọn — **tôi (thông qua Agent tự đề xuất và tôi đồng ý hướng an toàn hơn)** chọn phương án thêm rule allow-list cụ thể (`!outputs/*.csv`) thay vì xoá rule chặn gốc, để tránh vô tình commit nhầm file dataset 300MB (`hackernoon_subset.csv`) lên GitHub.

**Giải pháp scale lên 350MB (~100,000 bài báo):**

Bottleneck đầu tiên **không phải Neo4j** mà là **thông lượng LLM extraction**. Ước tính: 100k bài × vài chunk/bài ≈ vài trăm nghìn chunk; ở batch size 4/request và rate-limit free-tier (~30 RPM), chạy tuần tự đơn luồng sẽ mất hàng chục ngày — hoàn toàn không khả thi.

Giải pháp kiến trúc:
1. **Song song hoá có kiểm soát** (worker pool đa luồng, đã áp dụng ở quy mô nhỏ trong lab này cho bước extraction/evaluation) kết hợp nhiều API key luân phiên để tăng thông lượng mà không vượt rate-limit từng key.
2. **Pre-filter rẻ trước khi gửi LLM:** dùng NER nhẹ (spaCy) lọc chunk có ít nhất 2 thực thể tiềm năng trước khi tốn lượt gọi LLM đắt tiền cho NER+RE đầy đủ.
3. **Entity Resolution đổi FAISS `IndexFlatIP` (O(N²) khi search) sang HNSW** kèm blocking theo ký tự đầu/loại thực thể, vì ở quy mô hàng trăm nghìn mention, Flat index sẽ không còn khả thi về thời gian.
4. **Neo4j:** dùng `apoc.periodic.iterate` cho các thao tác ghi lớn, tăng heap size, cân nhắc self-hosted thay vì Aura Free (giới hạn tài nguyên).
5. Ở quy mô này, chính sách **Super-node cap (degree > 100)** mới thực sự phát huy tác dụng — ở quy mô lab (400 chunk), độ phổ biến node cao nhất còn thấp hơn ngưỡng 100, nên cơ chế cắt tỉa cần được kiểm chứng bằng cách hạ tạm ngưỡng thử nghiệm (xem mục 3).
