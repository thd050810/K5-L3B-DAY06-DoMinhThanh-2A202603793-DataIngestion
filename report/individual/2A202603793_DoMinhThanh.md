# Báo cáo cá nhân — Day 6

**Học viên:** Đỗ Minh Thanh  
**MSSV:** 2A202603793  
**Lớp:** L3B  
**Ngày lập báo cáo:** 14/11/2026

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** Xây dựng kho tri thức truy xuất từ sách *Machine Learning cơ bản* của Vũ Hữu Tiệp, bản cập nhật 27/03/2018. Nguồn được ghi trong [manifest](../../starter/data/manifest.json), file local `starter/data/book_ML_color.pdf`; địa chỉ ebook được ghi trong tài liệu là `https://machinelearningcoban.com/ebook/`. PDF có 400 trang vật lý, chỉ chọn **34 trang nội dung từ 76–109**, tương ứng trang in 64–97: chương 5 ở trang 76–82, chương 6 ở trang 83–94, chương 7 ở trang 95–102 và chương 8 ở trang 103–109. Số trang dùng trong pipeline/gold bắt đầu từ 1 theo PDF; không tính bìa, mục lục hoặc trang trống. Trang 94, 102, 109 có mục đọc thêm nhưng vẫn chứa nội dung chuyên môn nên được giữ. Văn bản chuẩn hóa có **76.302 ký tự**. Sách có thông báo bản quyền; manifest chưa có đầy đủ xác nhận quyền xử lý, tái phân phối và ngày truy cập. Đây là phần hồ sơ còn thiếu, cần xác minh trước khi chia sẻ dữ liệu gốc hoặc các trích đoạn.

- **Chuẩn hóa, lỗi và giới hạn:** Dùng `pypdf==6.21.4` trích xuất chữ riêng trên 34 trang đã chọn; chuẩn hóa Unicode NFC, CRLF/CR thành LF, gom khoảng trắng ngang và giới hạn dòng trống liên tiếp. Không dùng OCR. Khi chuyển từ `pypdf 4.3.1` sang `6.21.4`, khoảng trắng quanh ký hiệu/thuật ngữ thay đổi, từng làm kiểm chứng evidence lỗi ở q02; evidence của 12 câu đã được đồng bộ trước hai run, xác nhận nội dung bỏ khoảng trắng không đổi. Loader ban đầu quét cả PDF và gặp cảnh báo trang trống/form XObject; bản dùng cho hai run chỉ trích xuất trang được chọn. **0/34 trang có ranh giới đoạn `\n\n`**, nên chiến lược theo đoạn chủ yếu rơi vào nhánh cắt khối tối đa 500 ký tự. Còn lỗi dính chữ, công thức, header/footer và cắt giữa câu.

- **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí:** Dùng `sentence-transformers/all-MiniLM-L6-v2`, inference local trên CPU qua `sentence-transformers==5.7.0`, batch size 32; vector **384 chiều**, chuẩn hóa L2, cùng model cho document/query. Chưa lưu revision cụ thể của model. Vector store là PostgreSQL 16.16 + pgvector 0.8.7 theo kiểm tra hạ tầng trong phiên làm việc; tìm kiếm exact cosine, lấy top-3, filter JSONB trước LIMIT. Mỗi run/strategy có namespace riêng; `upsert` thay toàn bộ chunk của document trong transaction để tránh giữ chunk cũ, nhưng artifacts chưa có kiểm tra nạp lặp trực tiếp trên database. **Chi phí API bằng 0**, chưa tính điện, thiết bị, băng thông. Lệnh tái lập chạy từ gốc repo sau khi setup theo [starter/README.md](../../starter/README.md):

```bash
cd starter
python -m src.run --split dev --fixed-size 500 --overlap 50 --paragraph-size 500 --batch-size 32 --device cpu
python -m src.run --split test --fixed-size 500 --overlap 50 --paragraph-size 500 --batch-size 32 --device cpu
```

- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn:** Mỗi chunk giữ `document_id`, `source`, `page` để truy vết. Metadata bổ sung gồm `title`, `author`, `language`, `year`, `source_url`, `chapter`, `chapter_title`, `section`, `printed_page`, `document_type`, `is_empty`. Tên sách/tác giả lấy từ bìa; năm từ trang PDF 2; chương và section lấy từ tiêu đề/phạm vi trang; `printed_page` đối chiếu số trang in, trong phạm vi này bằng trang PDF trừ 12. `document_type`, `is_empty` do loader xác định. Trường lọc `section` có bốn giá trị: `basic-concepts`, `feature-engineering`, `linear-regression`, `overfitting`, giúp giới hạn truy xuất khi câu hỏi đã nêu chương. Filter lấy từ phạm vi câu hỏi, không lấy gold page/đáp án. `chapter`, `printed_page`, `year` là số nguyên, `section`, `language` là chuỗi, `is_empty` là boolean; năm/ngôn ngữ giống nhau toàn corpus nên không giúp phân biệt phạm vi.

## Cấu hình và benchmark

- **A / B / C (tham số cụ thể, khác biệt duy nhất B–C):** A chia cửa sổ **500 ký tự, overlap 50**, không lọc metadata, tạo **184 chunk**. B chia theo đoạn, tối đa **500 ký tự**, cắt đoạn quá dài, **không overlap**, không lọc, tạo **172 chunk**. C dùng nguyên chunk/vector của B, chỉ thêm `question.filter.section` lúc tìm kiếm; câu không có filter chạy như B. Như vậy B–C chỉ thay điều kiện lọc. A–B thay cách chia và overlap, nhưng do mất ranh giới đoạn trong PDF, chưa thể coi B là phép giữ trọn ý đoạn văn. Tổng ký tự các chunk A là 83.802, B/C là 76.302; độ dài nhỏ nhất/trung vị/lớn nhất lần lượt A: 69/500/500, B/C: 5/500/500 ký tự.

- **≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence:** [Benchmark](../../starter/data/benchmark.json) có **20 câu**, split cố định: **dev q01–q15** thuộc chương 5–7, **test q16–q20** thuộc chương 8. Dev có filter ở q03, q07, q09, q12; test ở q16, q19. Cả hai split truy xuất trên cùng corpus 34 trang. Mỗi câu có câu hỏi, đáp án chuẩn, `gold_source_id`, `gold_page`, `evidence_quote`; q02 có alternative gold trang 95 ngoài gold chính trang 77. Evidence được đối chiếu với văn bản chuẩn hóa đúng nguồn/trang; ít nhất một gold phải phù hợp filter. Một hit yêu cầu đúng document ID, trang và chứa trọn evidence sau chuẩn hóa NFC/khoảng trắng, hoặc khớp một alternative gold. Không ghép nhiều chunk để tính hit. Hit@1 xét hạng đầu; Evidence Hit@3 xét ít nhất một hit trong top-3. Đây là metric retrieval; pipeline chưa sinh/chấm đáp án LLM. Codex hỗ trợ soạn benchmark, code, phân tích và trình bày báo cáo; kết quả được kiểm tra bằng evidence, summary, trace và snapshot, không thay cho audit thủ công của học viên.

- **Tiêu chí chọn chiến lược đã chốt trước khi xem test:** Manifest ghi ưu tiên **Evidence Hit@3 trên dev → Hit@1 → thời gian ingest và số chunk**. Không chọn lại cấu hình theo test. Dev B/C hòa độ chính xác và số chunk; C tái sử dụng embedding B nên thời gian ingest không so trực tiếp như dựng độc lập. Vì vậy tiêu chí ban đầu chưa xác định duy nhất C là chiến lược thắng. [Experiment log](../../artifacts/experiment_log.csv) được bổ sung hậu nghiệm từ kết quả đã lưu, không phải nhật ký giả thuyết trước khi chạy.

## Kết quả

Số liệu lấy từ run dev `20261114T041647Z_62a043b1` và run test `20261114T041801Z_c8a613be`; không chạy lại benchmark để viết báo cáo này. Các phân số được đối chiếu với summary và trace của từng run.

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A | Dev | 5/15 (33,33%) | 6/15 (40,00%) | 184 | Ingest 3,859 s; embedding 3,479 s | API: 0 |
| B | Dev | 5/15 (33,33%) | 6/15 (40,00%) | 172 | Ingest 3,224 s; embedding 2,823 s | API: 0 |
| C | Dev | 5/15 (33,33%) | 6/15 (40,00%) | 172 | Ingest 0,093 s; embedding 0,000 s* | API: 0 |
| A | Test | 1/5 (20,00%) | 1/5 (20,00%) | 184 | Ingest 4,321 s; embedding 4,116 s | API: 0 |
| B | Test | 1/5 (20,00%) | 2/5 (40,00%) | 172 | Ingest 4,000 s; embedding 3,861 s | API: 0 |
| C | Test | 1/5 (20,00%) | 2/5 (40,00%) | 172 | Ingest 0,098 s; embedding 0,000 s* | API: 0 |

\* C dùng lại embedding B; đây là chi phí tăng thêm, không phải thời gian dựng C độc lập. Embedding nằm trong thời gian ingest, không cộng hai giá trị. Ingest đo từ tạo/chọn chunk đến sau upsert, không gồm đọc PDF, tải/khởi tạo model, query embedding hoặc đánh giá.

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | Chưa chấm (10 mẫu đã chuẩn bị) | Chưa audit thủ công; tự động nhất quán snapshot test: 15/15 lượt chunk | Dev: 3/4; test: 2/2 (đối chiếu hậu nghiệm) | Chưa chấm 5 câu test; chưa có số câu 0/1/2 và trung bình |
| B | Chưa chấm (10 mẫu đã chuẩn bị) | Chưa audit thủ công; tự động nhất quán snapshot test: 15/15 lượt chunk | Dev: 3/4; test: 2/2 (đối chiếu hậu nghiệm) | Chưa chấm 5 câu test; chưa có số câu 0/1/2 và trung bình |
| C | Chưa chấm (10 mẫu đã chuẩn bị) | Chưa audit thủ công; tự động nhất quán snapshot test: 15/15 lượt chunk | Dev: 4/4; test: 2/2 | Chưa chấm 5 câu test; chưa có số câu 0/1/2 và trung bình |

Filter tuân thủ khi mọi chunk trả về thỏa filter của câu hỏi; kết quả rỗng không tính thành công. A/B không áp dụng filter nên tỷ lệ trên chỉ phản ánh kết quả có tình cờ đúng phạm vi hay không. Kiểm tra tự động metadata so source/page/metadata/text với `pages.json`; không chứng minh gán metadata đúng theo PDF và không thay thế tỷ lệ audit thủ công. Chưa có nhãn thì không ghi điểm 0 hoặc suy điểm coherence/grounding từ Hit@3.

**Đường dẫn trace, nhãn thủ công và cách chọn mẫu:**

- **Dev:** [summary](../../starter/results/20261114T041647Z_62a043b1/summary.json), [trace top-3](../../starter/results/20261114T041647Z_62a043b1/traces.jsonl), [config](../../starter/results/20261114T041647Z_62a043b1/config.json), [resources](../../starter/results/20261114T041647Z_62a043b1/resources.json), [mẫu coherence](../../starter/results/20261114T041647Z_62a043b1/coherence_review.json).
- **Test:** [summary](../../starter/results/20261114T041801Z_c8a613be/summary.json), [trace top-3](../../starter/results/20261114T041801Z_c8a613be/traces.jsonl), [config](../../starter/results/20261114T041801Z_c8a613be/config.json), [resources](../../starter/results/20261114T041801Z_c8a613be/resources.json), [mẫu coherence](../../starter/results/20261114T041801Z_c8a613be/coherence_review.json).
- **Chọn mẫu coherence:** cùng 10 vị trí đầu trang PDF 76–85, sắp theo document ID và số trang, áp dụng cho cả ba chiến lược; mỗi file có 30 mục và `coherence_score=null`. Hai file thuộc hai run cùng corpus, không phải 20 mẫu độc lập mỗi chiến lược.
- **Nhãn còn thiếu:** coherence 0/1/2 kèm lý do, audit metadata với nguồn, grounding toàn bộ top-3 của q16–q20. Trace có trường dành cho đánh giá thủ công nhưng chưa điền; chưa có bộ nhãn thủ công hoàn chỉnh.
- **Đối chiếu tự động:** [report_verification.json](../../artifacts/report_verification.json) lưu kiểm tra summary/trace và provenance; đây là kiểm tra nhất quán pipeline.

## Phân tích và kết luận

- **So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn:** Trên dev, A/B cùng Hit@1 5/15 và Hit@3 6/15 nhưng tập câu đúng khác nhau. Trên test, B thêm một hit@3 so với A (2/5 so với 1/5), còn Hit@1 cùng 1/5. B/C giảm 12 chunk (**6,52%**) và 7.500 ký tự embedding (**8,95%**) so với A. C nâng tuân thủ filter dev từ 3/4 lên 4/4 nhưng không tăng Hit@1/Hit@3 ở cả hai split; trên test, top-3 của hai câu có filter không đổi giữa B/C. Hai ví dụ từ trace dev:
  - **q14 — bias trick; nguồn `book-ml`, `book_ML_color.pdf`, trang PDF 98 (trang in 86):** A trả đúng trang ở hai hạng đầu nhưng không chunk nào trong toàn corpus A chứa trọn evidence, nên tăng top-k không sửa được lỗi theo metric hiện tại. B/C có chunk chứa đủ evidence trong corpus nhưng top-3 lại ở trang 77, 83, 86: lỗi xếp hạng. Ví dụ phân biệt mất evidence do biên chunk với không truy xuất được chunk phù hợp.
  - **q03 — vai trò tập kiểm thử; nguồn `book-ml`, `book_ML_color.pdf`, trang PDF 79 (trang in 67):** C lọc đúng `basic-concepts`, top-1 và top-3 ở đúng trang 79 nhưng evidence bị chia giữa các chunk, không chunk đơn lẻ nào chứa đủ. B còn trả hai chunk ngoài phạm vi chương. Filter sửa phạm vi nhưng chưa sửa được thiếu ngữ cảnh và biên chunk.

- **Chiến lược chọn, đánh đổi, điều kiện áp dụng:** Chọn **B làm cấu hình cơ sở vận hành** vì giữ độ chính xác dev như A với ít chunk/ký tự embedding hơn. Dùng **C khi câu hỏi/ngữ cảnh có section đáng tin**, vì cải thiện tuân thủ phạm vi trên dev. Đây là lựa chọn có điều kiện; chưa có bằng chứng C tốt nhất tổng quát hoặc thắng duy nhất theo tiêu chí đã chốt. A vẫn có câu tốt hơn B, chẳng hạn test q19; không chọn B/C chỉ vì Hit@3 test cao hơn. Metadata filter không thay thế chất lượng chunk và ranking. Thiếu audit coherence/grounding nên kết luận chất lượng mới dựa chủ yếu trên retrieval tự động và kiểm tra phạm vi.

- **Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí:** Corpus chỉ một sách, 34/400 trang và bốn chương; các chương có khái niệm/tham chiếu chung nên split theo chương chưa bảo đảm độc lập hoàn toàn. Test chỉ 5 câu, mỗi câu làm tỷ lệ đổi 20 điểm phần trăm; chưa đủ kết luận ưu thế tổng quát. Exact evidence nhạy với parser, dấu câu và biên chunk; đúng trang hoặc ý tương đương chưa chắc được tính hit. Mẫu coherence tập trung trang đầu và chưa có nhãn thủ công; metadata/grounding cũng chưa audit. Model dùng cho corpus tiếng Việt nhưng chưa có thí nghiệm đối chứng model đa ngữ. Mỗi cấu hình/split chỉ đo một lần theo thứ tự A → B → C, chưa kiểm soát cache/warm-up; C dùng lại vector B nên thời gian không so độc lập. Chưa đo peak RAM, token count, chi phí thiết bị/điện hoặc khóa đầy đủ model revision/dependency. Quyền sử dụng/ngày truy cập, repo và commit chốt bài còn cần bổ sung.

- **Nếu làm lại hoặc mở rộng, cải tiến gì?** Hoàn tất nhãn coherence, metadata và grounding với nguồn trước khi kết luận đủ năm khía cạnh. Trên dev, ưu tiên phục hồi đoạn/heading, chia theo câu, rà chunk quá ngắn và kiểm tra riêng ảnh hưởng overlap; sau đó thử model phù hợp tiếng Việt, hybrid retrieval hoặc reranking bằng thí nghiệm riêng. Giữ baseline, corpus và định nghĩa metric cố định; nếu thêm đánh giá evidence đa chunk/ngữ nghĩa thì báo thành metric bổ sung. Khóa revision model/dependency, ghi cấu hình máy, lặp đo và tách chi phí dựng độc lập với chi phí tái sử dụng. Giữ test hiện tại để báo cáo; nếu dùng lỗi test để chỉnh pipeline, ghi rõ và xây tập test độc lập mới.
