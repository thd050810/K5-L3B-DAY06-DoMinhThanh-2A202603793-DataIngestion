# Starter · PostgreSQL + pgvector

Pipeline local cho lab cá nhân: đọc PDF, chia chunk, embedding bằng MiniLM, lưu vào PostgreSQL + pgvector và đánh giá truy xuất. Cần tự chuẩn bị manifest và bộ câu hỏi có evidence đã kiểm chứng.

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf, dotenv và sentence-transformers
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc PDF, chuẩn hóa Unicode và khoảng trắng
│   ├── chunking.py        # fixed-size và theo đoạn văn
│   ├── embeddings.py      # MiniLM local, vector 384 chiều
│   ├── store.py           # lưu vector và tìm cosine có filter
│   ├── evaluation.py      # Hit@1, Evidence Hit@3 và trace
│   └── run.py             # CLI nối pipeline A/B/C
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Chọn embedding model

`src/embeddings.py` dùng model local [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), tạo vector **384 chiều**, chuẩn hóa L2 cho cosine similarity. `sentence-transformers` đã có trong `requirements.txt`. Lần đầu cần mạng để tải model vào cache Hugging Face; việc tính embedding chạy trên máy, không cần API key.

Chạy từ thư mục `starter/` sau khi cài dependencies:

```python
from src.embeddings import MyEmbedder

embedder = MyEmbedder(batch_size=32, device="cpu")
vectors = embedder.embed_documents(["Machine learning learns patterns from data."])
query_vector = embedder.embed_query("What is machine learning?")
print(embedder.dimensions, len(vectors), len(query_vector))
```

Mặc định chạy CPU; có thể truyền `device="mps"` trên máy Apple hỗ trợ hoặc `device="cuda"` khi có GPU tương thích. Dùng cùng instance/model cho document và query. Model dành cho tiếng Anh và tự cắt nội dung vượt **256 word pieces**; cần chọn kích thước chunk phù hợp. `MyEmbedder()` dùng model mặc định; nếu muốn đọc `EMBEDDING_MODEL` từ `.env`, hãy load biến môi trường rồi truyền vào `model_name`.

Kiểm tra từ thư mục gốc repo: `python -m unittest discover -s starter/tests -v`. Test embedding chạy model thật và cần tải model nếu chưa có cache.

## 3. Chuẩn bị dữ liệu

Copy `data/manifest.example.json` thành `data/manifest.json`. Mỗi phần tử mô tả một PDF:

- `document_id`: ID duy nhất; `path`: đường dẫn tương đối với file manifest; `source`: tên/URL để truy vết.
- `content_pages`: liệt kê số trang vật lý, bắt đầu từ 1, đã kiểm chứng là nội dung hữu ích. Tổng tối thiểu 30 trang; loại bìa, mục lục, trang trống và tài liệu tham khảo. Mẫu để trống vì chưa kiểm chứng PDF.
- `metadata`: thuộc tính có bằng chứng trong nguồn, áp dụng toàn tài liệu.
- `page_metadata`: metadata riêng theo trang, ví dụ `{"12": {"section": "linear-algebra"}}`. Giá trị ở trang ghi đè metadata tài liệu. `document_type` và `is_empty` do loader xác định.

Chọn metadata có ít nhất hai phạm vi phân biệt được theo [GUIDE.md](../GUIDE.md); kiểm chứng thủ công, không gán giá trị giả để đủ filter.

Copy `data/benchmark.example.json` thành `data/benchmark.json`, thay placeholder và thêm tối thiểu **15 câu dev / 5 câu test**. Mỗi câu có ID duy nhất, câu hỏi, đáp án, `gold_source_id` trùng `document_id`, số trang và evidence nguyên văn. Ít nhất ba câu có filter, gồm một câu test. Mẫu chỉ minh họa schema, không phải benchmark hoàn chỉnh.

Filter dùng `section`, `year`, `version`, `category`, `audience`, `language`, `effective_date` hoặc `document_type`, với giá trị scalar khớp cả kiểu dữ liệu. Filter phải được biết trước từ câu hỏi/ngữ cảnh, không suy từ đáp án hoặc gold. Nếu có nguồn/trang hợp lệ khác, thêm vào `alternative_gold` các object có `gold_source_id`, `gold_page`, `evidence_quote`. Evidence của từng alternative phải tồn tại ở đúng nguồn/trang.

## 4. Chạy và xem kết quả

Sau khi database hoạt động và đã chuẩn bị hai file JSON, chạy từ `starter/`:

```bash
python -m src.run --split dev
```

Mặc định A chia cửa sổ 500 ký tự, overlap 50; B theo đoạn tối đa 500 ký tự; C dùng nguyên chunk/vector của B, thêm filter của câu hỏi. Câu không có filter thì C tìm như B. Chọn tham số trên dev, cố định cấu hình rồi mới chạy test:

```bash
python -m src.run --split test --fixed-size 500 --overlap 50 --paragraph-size 500
```

`--split all` chạy cả hai tập, chỉ dùng sau khi cố định cấu hình. Xem tùy chọn bằng `python -m src.run --help`. Cấu hình database đọc từ `starter/.env` hoặc biến môi trường; không được ghi vào kết quả.

Mỗi lần chạy tạo `results/<run_id>/` và namespace database riêng để không trộn dữ liệu/model giữa thí nghiệm. Trong một namespace, `store.upsert` nhận **toàn bộ chunk hiện tại của mỗi tài liệu được cập nhật**, thay chunk cũ trong cùng transaction; không gọi từng batch chunk cho cùng tài liệu. Thư viện truy xuất bằng exact cosine với filter JSONB trước khi lấy top-3.

Kết quả gồm:

- `config.json`, `corpus.json`, `benchmark.json`, `pages.json`: cấu hình, hash đầu vào, manifest và dữ liệu chuẩn hóa để tái lập.
- `chunks_A.json`, `chunks_B.json`, `chunks_C.json`: chunk và metadata từng chiến lược.
- `traces.jsonl`: top-3, score, nguồn/trang, gold, filter đã áp dụng, evidence match và thời gian tìm kiếm.
- `summary.json`: Hit@1 và Evidence Hit@3 theo chiến lược/dev-test, kèm tử/mẫu; tỷ lệ tuân thủ filter, `null` nếu không có câu filter. Có filter nhưng không trả kết quả được tính là không thành công.
- `resources.json`: số chunk, tổng ký tự, thời gian embedding/ingest; C tái sử dụng vector B. Chi phí API bằng 0, không bao gồm điện/phần cứng.
- `coherence_review.json`: mẫu cùng 10 vị trí nguồn đầu đã sắp theo document ID và trang cho cả ba chiến lược. Điền coherence 0–2 và lý do. Trong trace, điền grounding 0–2 cho câu test và kiểm tra metadata thủ công; code không suy các điểm này từ string match.

Hit tự động yêu cầu đúng document ID, trang và chứa trọn evidence sau chuẩn hóa NFC/khoảng trắng; evidence nằm rải qua nhiều chunk không được tính. Bằng chứng tương đương về ngữ nghĩa cần người chấm xác nhận riêng. Việc có đủ chữ không tự chứng minh đáp án/evidence có chất lượng: vẫn cần chấm theo [BENCHMARK.md](../BENCHMARK.md).

Chạy toàn bộ test từ gốc repo. Hai test tích hợp database sẽ skip nếu chưa đặt `TEST_DATABASE_URL` trỏ tới database pgvector dành cho test. Test CLI có dùng model thật và tạo dữ liệu trong namespace riêng.

Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.
