# TextVQA Feature Restoration — Phase 2–7

Phục hồi đặc trưng ảnh suy thoái trong latent space của Qwen2.5-VL và đánh giá trên TextVQA/ST-VQA.

## 1. Máy chủ cần có

- Linux và GPU NVIDIA H200
- Python 3.10, 3.11 hoặc 3.12
- Git và Internet

## 2. Dữ liệu được tải ở đâu?

Chạy `python run_all.py` sẽ tự tải các Kaggle Dataset **Public** vào đúng cấu trúc:

```text
textvqa-feature-restore-phase2/
├── data/
│   ├── phase1/           # embechuangoan/textvqa-feature-restore-v1
│   ├── degraded/         # anhthu128/data-suy-thoai-text-vqa
│   ├── clean/            # dmytruto/textvqa
│   ├── stvqa_degraded/   # anhthu128/st-vqa-suy-thoi
│   └── stvqa_clean_from_huggingface/  # chương trình tự tạo
└── outputs/              # toàn bộ checkpoint, metric và biểu đồ
```

Kaggle Dataset công khai được tải ẩn danh bằng `kagglehub`, không cần username, API key hay đăng nhập. Cả bốn Dataset trên phải để **Public**. Nếu một Dataset còn Private, chương trình sẽ dừng và báo tên Dataset đó.

ST-VQA sạch được tải từ snapshot `vikhyatk/st-vqa` trên Hugging Face. Model `Qwen/Qwen2.5-VL-3B-Instruct` cũng tự tải từ Hugging Face và được cache để dùng lại.

Không đổi tên file bên trong Dataset và không sửa manifest CSV.

## 3. Chạy trên server

```bash
git clone https://github.com/ducdanh2304/textvqa-feature-restore-phase2.git
cd textvqa-feature-restore-phase2
python -m pip install -r requirements.txt
python run_all.py
```

Không cần truyền đường dẫn hoặc tham số. Phiên bản thư viện chỉ nằm trong `requirements.txt`; chương trình sẽ dừng ngay nếu môi trường cài sai phiên bản.

Nếu tiến trình bị ngắt, chạy lại:

```bash
python run_all.py
```

Không xóa `data/` hoặc `outputs/`; chương trình dùng lại dữ liệu, cache và checkpoint đã hoàn thành.

## 4. Kết quả chính

```text
outputs/
├── bootstrap_status.json
├── resolved_inputs.json
├── pipeline_state.json
├── final_summary.json
├── phase2_training/
├── phase3_baseline_evaluation/
├── phase4_identity/
├── phase5_token_gate/
├── phase6_question_gate/
└── phase7_answer_loss/
```

Xem `outputs/final_summary.json` để đọc kết quả tổng hợp. Mỗi phase có checkpoint, dự đoán, metric, log và biểu đồ trong thư mục tương ứng.

## 5. Khi chương trình báo lỗi

- `Dataset ... Public`: mở trang Dataset trên Kaggle, vào **Settings** và chuyển sang Public.
- `Môi trường không khớp requirements.txt`: cài lại đúng môi trường Python rồi chạy `python -m pip install -r requirements.txt`.
- Thiếu Internet: máy chủ phải truy cập được Kaggle và Hugging Face trong lần chạy đầu.
- Hết GPU memory: kiểm tra tiến trình khác đang sử dụng GPU.

