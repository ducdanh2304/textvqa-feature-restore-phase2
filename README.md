# Phục hồi đặc trưng ảnh suy thoái cho TextVQA — Giai đoạn 2 đến 7

Kho mã này chạy toàn bộ nghiên cứu bằng **một tệp Python duy nhất**:

```bash
python3.11 run_pipeline.py \
  --input-root /datastore/dungmt/textvqa/input \
  --output-root /datastore/dungmt/textvqa/output
```

Người chạy không cần sửa YAML, không cần khai báo biến môi trường và không cần chạy `run.sh`.
Chương trình tự tìm dữ liệu trong `--input-root`, kiểm tra cấu trúc, tải mô hình Qwen, chạy lần lượt Giai đoạn 2–7 và tiếp tục được khi bị ngắt.

## 1. Nghiên cứu này làm gì?

Mục tiêu là sửa **feature thị giác bên trong Qwen2.5-VL**, không tạo ảnh phục hồi ở mức pixel.

```text
Ảnh suy thoái → Vision Encoder → feature suy thoái → mô-đun phục hồi → feature đã sửa → Qwen trả lời
Ảnh sạch     → Vision Encoder → feature sạch ────────┘ dùng làm mục tiêu khi huấn luyện
```

| Giai đoạn | Việc thực hiện | Câu hỏi cần trả lời |
|---|---|---|
| 2 | Huấn luyện residual adapter bằng feature sạch/suy thoái | Có kéo feature suy thoái gần feature sạch được không? |
| 3 | Đánh giá `clean/degraded × base/adapter` | Adapter cơ sở có làm TextVQA chính xác hơn không? |
| 4 | Thêm identity loss trên feature sạch | Có giảm trường hợp sửa feature đang tốt thành xấu không? |
| 5 | Thêm gate riêng cho từng visual token | Có học được token nào cần sửa và sửa bao nhiêu không? |
| 6 | Cho gate nhìn thêm câu hỏi | Có ưu tiên đúng token liên quan đến câu hỏi không? |
| 7 | Tối ưu thêm bằng answer loss của Qwen đóng băng | Feature đã sửa có trực tiếp giúp câu trả lời đúng hơn không? |

Giai đoạn 3–7 đều báo cáo Official TextVQA Accuracy, ANLS, chênh lệch so với base, bootstrap 95% CI và win/tie/loss.

## 2. Cần tải những gì?

Cần đúng ba dataset và một mô hình:

| Thành phần | Liên kết | Dùng để làm gì? |
|---|---|---|
| Kết quả Giai đoạn 1 | [textvqa-feature-restore-v1](https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1) | 2.000 cặp train, 200 cặp dev và manifest |
| Ảnh suy thoái benchmark | [data-suy-thoai-text-vqa](https://www.kaggle.com/datasets/anhthu128/data-suy-thoai-text-vqa) | 1.000 ảnh `realistic_mix/L2` để đánh giá |
| Ảnh TextVQA sạch | [textvqa](https://www.kaggle.com/datasets/dmytruto/textvqa) | 1.000 ảnh sạch tương ứng để đánh giá |
| Mô hình | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision encoder và mô hình trả lời VQA |

Dataset Giai đoạn 1 đang để **Private**. Chủ dataset phải vào **Settings → Sharing** và thêm đúng tài khoản Kaggle của người chạy trước khi tải.

Không cần tải dataset `outputs.zip` của khảo sát cũ. Nó chỉ là kết quả tham khảo, không phải đầu vào huấn luyện.

## 3. Chuẩn bị máy H200 từ đầu

Yêu cầu: Linux, GPU NVIDIA H200, Python 3.10/3.11, Git và đủ dung lượng cho dữ liệu, Qwen, feature cache.

### Bước 1 — lấy mã nguồn

```bash
cd /datastore/dungmt
git clone https://github.com/ducdanh2304/textvqa-feature-restore-phase2.git
cd textvqa-feature-restore-phase2
```

### Bước 2 — tạo môi trường tự động

```bash
python3.11 run_pipeline.py --help
```

Lần đầu, tệp này tự tạo `.venv` và cài đúng thư viện. Những lần sau nó dùng lại môi trường đó.

### Bước 3A — thuận tiện nhất: tải tự động bằng Kaggle

Đăng nhập Kaggle một lần:

```bash
.venv/bin/kaggle auth login --no-launch-browser
```

Mở địa chỉ được in ra, đăng nhập, rồi dán mã xác thực vào terminal. Sau đó chạy:

```bash
python3.11 run_pipeline.py \
  --download-data \
  --input-root /datastore/dungmt/textvqa/input \
  --output-root /datastore/dungmt/textvqa/output
```

Tùy chọn `--download-data` tự tải và giải nén cả ba dataset. Nếu một thư mục đã có dữ liệu, chương trình giữ nguyên thư mục đó.

### Bước 3B — nếu tải bằng trình duyệt

1. Mở ba liên kết ở mục 2 và bấm **Download**.
2. Chuyển ba tệp ZIP lên máy H200.
3. Giải nén mỗi ZIP vào một thư mục con bất kỳ bên trong `/datastore/dungmt/textvqa/input`.

```text
/datastore/dungmt/textvqa/input/
├── phase1/       ← giải nén textvqa-feature-restore-v1 vào đây
├── degraded/     ← giải nén data-suy-thoai-text-vqa vào đây
└── clean/        ← giải nén dmytruto/textvqa vào đây
```

Tên và số tầng thư mục con không bắt buộc giống ví dụ. Chương trình tự dò file cần thiết. Không di chuyển riêng từng ảnh và không sửa đường dẫn trong CSV.

## 4. Cấu trúc dữ liệu mà chương trình tự tìm

```text
Giai đoạn 1/
└── textvqa_feature_restore_v1/
    ├── train_manifest.csv
    ├── dev_manifest.csv
    ├── benchmark_excluded_image_ids.csv
    └── data/
        ├── train/clean/       (2.000 ảnh)
        ├── train/degraded/    (2.000 ảnh)
        ├── dev/clean/         (200 ảnh)
        └── dev/degraded/      (200 ảnh)

Dataset suy thoái/
└── textvqa_realistic_v2/main/
    ├── final_main_manifest.csv             (1.000 dòng)
    └── images/realistic_mix/L2/             (1.000 ảnh PNG)

Dataset sạch/
└── .../train_images/                       (có đủ 1.000 image_id trong manifest)
```

Chương trình ghi đường dẫn đã tìm thấy vào `output/resolved_inputs.json`. Nếu tìm thấy 0 hoặc nhiều hơn 1 vị trí hợp lệ, nó dừng và liệt kê các vị trí để tránh dùng nhầm dataset.

## 5. Lệnh chạy chính

### Máy có Internet

```bash
python3.11 run_pipeline.py \
  --input-root /datastore/dungmt/textvqa/input \
  --output-root /datastore/dungmt/textvqa/output
```

Qwen được tải tự động từ Hugging Face ở revision đã khóa.

### Máy không có Internet

Chép snapshot đầy đủ của Qwen lên máy rồi chạy:

```bash
python3.11 run_pipeline.py \
  --input-root /datastore/dungmt/textvqa/input \
  --output-root /datastore/dungmt/textvqa/output \
  --model-path /datastore/dungmt/models/Qwen2.5-VL-3B-Instruct \
  --offline
```

### Chạy lại sau khi bị ngắt

Chạy lại đúng lệnh cũ. Chương trình giữ feature cache, checkpoint và prediction đã hoàn thành. Không xóa `output` nếu muốn tiếp tục.

### Chỉ chạy tiếp một số giai đoạn

```bash
python3.11 run_pipeline.py \
  --input-root /datastore/dungmt/textvqa/input \
  --output-root /datastore/dungmt/textvqa/output \
  --phases 4,5,6,7
```

Chương trình kiểm tra checkpoint giai đoạn trước trước khi chạy tiếp.

## 6. Kết quả nằm ở đâu?

```text
output/
├── resolved_inputs.json
├── pipeline_state.json
├── final_summary.json
├── phase2_training/
│   ├── cache/
│   ├── training/best_adapter.pt
│   └── verification/
├── phase3_baseline_evaluation/
│   ├── predictions.csv
│   ├── scored_predictions.csv
│   └── evaluation_report.json
├── phase4_identity/
│   ├── best_model.pt
│   ├── history.csv
│   └── evaluation/evaluation_report.json
├── phase5_token_gate/
├── phase6_question_gate/
└── phase7_answer_loss/
    ├── best_model.pt
    ├── history.csv
    └── evaluation/evaluation_report.json
```

Xem `final_summary.json` trước. Trong mỗi `evaluation_report.json`:

- `degraded_base`: Qwen gốc trên ảnh suy thoái;
- `degraded_adapter`: Qwen với mô-đun phục hồi;
- `mean_delta > 0`: adapter cải thiện trung bình;
- CI 95% nằm hoàn toàn trên 0: bằng chứng cải thiện rõ hơn;
- `wins/ties/losses`: số câu adapter làm tốt hơn, bằng nhau hoặc tệ hơn.

## 7. Quy tắc kết luận

1. Chỉ kết luận tốt hơn khi **Official TextVQA Accuracy trên ảnh suy thoái tăng**.
2. Feature MSE/cosine tốt hơn chỉ là chỉ báo trung gian.
3. So sánh trên cùng 1.000 câu hỏi, cùng model revision, prompt và cách giải mã.
4. Luôn xem clean arm để phát hiện negative restoration.
5. ANLS là chỉ số bổ sung; Official TextVQA Accuracy là chỉ số chính.

## 8. Lỗi thường gặp

### `Không tìm thấy bộ dữ liệu...`

Dataset chưa giải nén đầy đủ hoặc đặt ngoài `--input-root`. Đọc các đường dẫn trong lỗi rồi chuyển cả thư mục dataset vào `--input-root`.

### Tìm thấy nhiều vị trí

Cùng một dataset bị giải nén hai lần. Giữ một bản duy nhất trong `--input-root`.

### Kaggle báo không có quyền

Tài khoản đang đăng nhập chưa được chia sẻ dataset Giai đoạn 1. Chủ dataset cần thêm tài khoản đó trong **Settings → Sharing**.

### Hết bộ nhớ GPU

Kiểm tra không có tiến trình khác chiếm GPU. Cấu hình chính dành cho một H200. `--allow-non-h200` chỉ dùng chạy thử, không dùng làm kết quả chính.

### Xem tiến độ

Mở `pipeline_state.json`, `history.csv` và `predictions.csv`. Các tệp này được cập nhật trong quá trình chạy.
