# Phục hồi đặc trưng ảnh suy thoái cho TextVQA — Giai đoạn 2 đến 7

Kho mã này chạy toàn bộ nghiên cứu bằng **một tệp Python duy nhất** sau khi cài môi trường:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Người chạy không cần sửa YAML, không cần khai báo biến môi trường và không cần chạy `run.sh`.
Phiên bản Python và toàn bộ phiên bản thư viện nằm trong `requirements.txt`; `run_all.py` không tự cài package.
Chương trình tự dò thư mục cha chứa dữ liệu, kiểm tra dữ liệu, tải mô hình Qwen,
chạy lần lượt Giai đoạn 2–7 và tiếp tục được khi bị ngắt.

Các vị trí mặc định:

```text
textvqa-feature-restore-phase2/
├── data/       ← toàn bộ dataset tự tải vào đây
└── outputs/    ← toàn bộ kết quả được ghi vào đây
```

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

Giai đoạn 3–7 đều đánh giá trên cả TextVQA và ST-VQA, báo cáo metric chính, chênh lệch so với base, bootstrap 95% CI và win/tie/loss.

## 2. Cần tải những gì?

Cần các nguồn dữ liệu sau và một mô hình:

| Thành phần | Liên kết | Dùng để làm gì? |
|---|---|---|
| Kết quả Giai đoạn 1 | [textvqa-feature-restore-v1](https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1) | 2.000 cặp train, 200 cặp dev và manifest |
| Ảnh suy thoái benchmark | [data-suy-thoai-text-vqa](https://www.kaggle.com/datasets/anhthu128/data-suy-thoai-text-vqa) | 1.000 ảnh `realistic_mix/L2` để đánh giá |
| Ảnh TextVQA sạch | [textvqa](https://www.kaggle.com/datasets/dmytruto/textvqa) | 1.000 ảnh sạch tương ứng để đánh giá |
| Ảnh ST-VQA suy thoái | [st-vqa-suy-thoi](https://www.kaggle.com/datasets/anhthu128/st-vqa-suy-thoi) | Khoảng 5.000 ảnh `realistic_mix/L2` để kiểm tra ngoài miền |
| Ảnh ST-VQA sạch | [vikhyatk/st-vqa](https://huggingface.co/datasets/vikhyatk/st-vqa) | Nguồn ảnh sạch tương ứng; chương trình tự lấy đúng ảnh theo manifest |
| Mô hình | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision encoder và mô hình trả lời VQA |

Dataset Giai đoạn 1 đang để **Private**. Chủ dataset phải vào **Settings → Sharing** và thêm đúng tài khoản Kaggle của người chạy trước khi tải.

Không cần tải dataset `outputs.zip` của khảo sát cũ. Nó chỉ là kết quả tham khảo, không phải đầu vào huấn luyện.

ST-VQA được giữ làm **benchmark ngoài miền**, không trộn vào 2.000 cặp train. Nhờ vậy kết quả ST-VQA đo khả năng tổng quát sang một bộ dữ liệu khác thay vì đo việc mô hình ghi nhớ dữ liệu huấn luyện. Metric chính của ST-VQA là ANLS.

## 3. Chuẩn bị máy H200 từ đầu

Yêu cầu: Linux, GPU NVIDIA H200 (kể cả H200 MIG khoảng 33 GB), Python 3.10/3.11/3.12, Git và đủ dung lượng cho dữ liệu, Qwen, feature cache. Cấu hình hiện tại đã được chỉnh theo máy của trường: Python 3.12, H200 MIG khoảng 33 GB và NVIDIA driver hỗ trợ CUDA 13.0. PyTorch dùng CUDA runtime riêng nên không cần trùng đúng số phiên bản CUDA của driver.

### Bước 1 — lấy mã nguồn

```bash
git clone https://github.com/ducdanh2304/textvqa-feature-restore-phase2.git
cd textvqa-feature-restore-phase2
```

### Bước 2 — kích hoạt môi trường và cài thư viện

```bash
source /đường/dẫn/venv/bin/activate
python -m pip install -r requirements.txt
```

### Bước 3 — chạy toàn bộ thí nghiệm

```bash
python run_all.py
```

Lần đầu, chương trình sẽ:

1. Dò dataset từ `data/` cạnh repository, `data/`, `input/`, `datasets/` ở thư mục cha.
2. Nếu thiếu Dataset Kaggle, dừng và báo chính xác thành phần cần chép lên server; chương trình không đăng nhập hoặc tự tải Kaggle.
3. Tự tải ST-VQA sạch và Qwen từ Hugging Face.
4. Chạy Giai đoạn 2–7.
5. Ghi kết quả vào `outputs/` cạnh `run_all.py`.

Ngay khi khởi động, chương trình tạo `outputs/bootstrap_status.json`. Tệp này ghi Git commit, Python, đường dẫn output và giai đoạn hiện tại. Sau khi nạp PyTorch, tệp còn ghi tên GPU và dung lượng GPU nhìn thấy. Vì vậy kể cả khi cài thư viện hoặc pipeline lỗi, vẫn có trạng thái để kiểm tra.

Dataset Giai đoạn 1 phải được tải trước bởi tài khoản đã được chia sẻ quyền, sau đó chép và giải nén lên server cùng các Dataset còn lại.

### Trường hợp đã tải dữ liệu thủ công

1. Mở bốn liên kết Kaggle ở mục 2 và bấm **Download**.
2. Chuyển bốn tệp ZIP lên máy H200.
3. Giải nén mỗi ZIP vào một thư mục con bất kỳ bên trong thư mục `data/` của repository.

```text
textvqa-feature-restore-phase2/data/
├── phase1/       ← giải nén textvqa-feature-restore-v1 vào đây
├── degraded/     ← giải nén data-suy-thoai-text-vqa vào đây
├── clean/        ← giải nén dmytruto/textvqa vào đây
└── stvqa_degraded/ ← giải nén st-vqa-suy-thoi vào đây
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

ST-VQA suy thoái/
└── stvqa_realistic_v1_5k/main/
    ├── final_main_manifest.csv
    └── images/realistic_mix/L2/             (khoảng 5.000 ảnh)

ST-VQA sạch do chương trình tạo/
└── stvqa_clean_from_huggingface/stvqa_source_v1/images/
```

Chương trình ghép ST-VQA sạch bằng `source_row_index`, sau đó đối chiếu lại câu hỏi và danh sách đáp án với manifest suy thoái. Nếu không khớp, chương trình dừng thay vì đánh giá trên cặp ảnh sai. Các đường dẫn cuối cùng được ghi vào `output/resolved_inputs.json`.

## 5. Lệnh chạy chính

```bash
python run_all.py
```

Máy của trường có Internet nên Qwen và ST-VQA sạch được tải tự động ở đúng revision đã khóa.

### Chạy lại sau khi bị ngắt

Chạy lại đúng hai dòng trên. Chương trình giữ feature cache, checkpoint và prediction đã hoàn thành. Không xóa thư mục `outputs/` nếu muốn tiếp tục.

## 6. Kết quả nằm ở đâu?

```text
outputs/
├── resolved_inputs.json
├── pipeline_state.json
├── final_summary.json
├── phase2_training/
│   ├── cache/
│   ├── training/best_adapter.pt
│   ├── verification/
│   └── visualizations/
│       ├── feature_before_after_scatter.png
│       ├── feature_cosine_delta.png
│       └── feature_diagnostics.csv
├── phase3_baseline_evaluation/
│   ├── textvqa/
│   │   ├── predictions.csv
│   │   ├── per_question_comparison_degraded.csv
│   │   ├── visualizations/
│   │   │   ├── win_tie_loss.png
│   │   │   ├── score_delta_histogram.png
│   │   │   ├── base_vs_adapter_scatter.png
│   │   │   └── qualitative_examples.png
│   │   └── evaluation_report.json
│   └── stvqa/
│       ├── predictions.csv
│       └── evaluation_report.json
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

Mỗi Giai đoạn 3–7 có hai thư mục đánh giá riêng: `textvqa/` và `stvqa/`. Xem `final_summary.json` trước. Trong mỗi `evaluation_report.json`:

- `degraded_base`: Qwen gốc trên ảnh suy thoái;
- `degraded_adapter`: Qwen với mô-đun phục hồi;
- `mean_delta > 0`: adapter cải thiện trung bình;
- CI 95% nằm hoàn toàn trên 0: bằng chứng cải thiện rõ hơn;
- `wins/ties/losses`: số câu adapter làm tốt hơn, bằng nhau hoặc tệ hơn.

TextVQA dùng Official TextVQA Accuracy làm metric chính. ST-VQA dùng ANLS làm metric chính; không lấy TextVQA soft accuracy để kết luận cho ST-VQA.

Mỗi Giai đoạn 2, 4, 5, 6 và 7 có `visualizations/feature_diagnostics.csv` cùng biểu đồ so sánh feature trước/sau. Giai đoạn có gate còn có `gate_distribution.png`. Các Giai đoạn 4–7 có thêm `visualizations/training_curves.png`.

Mỗi lần đánh giá ở Giai đoạn 3–7 tự tạo:

- `per_question_comparison_clean.csv` và `per_question_comparison_degraded.csv`: câu hỏi, đáp án tham chiếu, câu trả lời base/adapter, điểm và chênh lệch;
- `visualizations/win_tie_loss.png`: adapter sửa đúng, không đổi và làm sai bao nhiêu câu;
- `visualizations/score_delta_histogram.png`: mức tăng/giảm điểm của từng câu;
- `visualizations/base_vs_adapter_scatter.png`: so sánh điểm trước/sau theo từng câu;
- `visualizations/qualitative_examples.png`: ảnh sạch, ảnh suy thoái đầu vào, câu trả lời và điểm trước/sau cho các trường hợp tiêu biểu.

Nghiên cứu phục hồi feature ẩn nên không có ảnh pixel “đã phục hồi”. Ảnh định tính luôn ghi rõ ảnh suy thoái là đầu vào thật; hiệu quả phục hồi được biểu diễn bằng thay đổi feature và câu trả lời VQA.

## 7. Quy tắc kết luận

1. Chỉ kết luận tốt hơn khi **Official TextVQA Accuracy trên ảnh suy thoái tăng**.
2. Feature MSE/cosine tốt hơn chỉ là chỉ báo trung gian.
3. So sánh trên cùng 1.000 câu hỏi, cùng model revision, prompt và cách giải mã.
4. Luôn xem clean arm để phát hiện negative restoration.
5. ANLS là chỉ số bổ sung; Official TextVQA Accuracy là chỉ số chính.

## 8. Lỗi thường gặp

### `Không tìm thấy bộ dữ liệu...`

Dataset chưa tải hoặc giải nén đầy đủ. Chạy lại `python run_all.py`; dữ liệu đã hoàn thành sẽ được giữ lại.

### Tìm thấy nhiều vị trí

Cùng một dataset bị giải nén hai lần. Giữ một bản duy nhất trong thư mục `data/`.

### Kaggle báo không có quyền

Tài khoản đang đăng nhập chưa được chia sẻ dataset Giai đoạn 1. Chủ dataset cần thêm tài khoản đó trong **Settings → Sharing**.

### Hết bộ nhớ GPU

Kiểm tra không có tiến trình khác chiếm GPU. Cấu hình chính dành cho một H200. `--allow-non-h200` chỉ dùng chạy thử, không dùng làm kết quả chính.

### Xem tiến độ

Mở `pipeline_state.json`, `history.csv` và `predictions.csv`. Các tệp này được cập nhật trong quá trình chạy.

## 9. Đánh giá nhanh checkpoint Phase 2 trên Kaggle T4 x2

Notebook [`notebooks/stvqa_phase3_t4x2.ipynb`](notebooks/stvqa_phase3_t4x2.ipynb) đánh giá checkpoint đã huấn luyện trên 500 câu ST-VQA cố định. Notebook không huấn luyện lại adapter.

Trước khi chạy:

1. Chọn accelerator **GPU T4 x2** và bật Internet.
2. Add Input dataset `anhthu128/st-vqa-suy-thoi`.
3. Add Input một dataset chứa đúng một file `best_adapter.pt`.
4. Chạy lần lượt toàn bộ cell.

Notebook đánh giá `clean/degraded × base/adapter` và xuất:

- ANLS làm metric chính;
- Official VQA Accuracy, Exact Match, BLEU-1, BLEU-2, BLEU-3 và BLEU-4 làm metric phụ;
- bootstrap CI 95% và win/tie/loss theo từng metric;
- câu trả lời và điểm của từng câu;
- biểu đồ tổng hợp và ví dụ adapter sửa đúng/làm sai;
- `/kaggle/working/stvqa_phase3_evaluation.zip` để tải hoặc lưu thành Kaggle Dataset.
