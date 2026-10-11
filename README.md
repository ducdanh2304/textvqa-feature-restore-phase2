# Giai đoạn 2: huấn luyện mô-đun phục hồi đặc trưng cho TextVQA

Kho mã này là bước tiếp theo sau thí nghiệm nền [textvqa-qwen-baseline](https://github.com/ncat218/textvqa-qwen-baseline). Thí nghiệm trước đo Qwen2.5-VL-3B trên 1.000 câu hỏi với ảnh sạch và ảnh suy thoái. Giai đoạn 2 này **không chạy lại bộ 1.000 ảnh**, mà chỉ huấn luyện mô-đun phục hồi trên tập huấn luyện và tập phát triển đã tách độc lập ở Giai đoạn 1.

Quy trình gồm:

1. Kiểm tra toàn bộ dữ liệu và nguồn gốc thí nghiệm.
2. Trích đặc trưng thị giác đóng băng của Qwen đúng một lần cho 4.400 ảnh.
3. Huấn luyện duy nhất mô-đun MLP có kết nối phần dư trên 2.000 cặp huấn luyện.
4. Chọn trọng số tốt nhất bằng hàm mất mát đặc trưng trên 200 cặp phát triển.
5. Nạp lại trọng số tốt nhất và kiểm tra một cặp đặc trưng phát triển.

**Giai đoạn 2 không tạo điểm độ chính xác TextVQA. Phải sang Giai đoạn 3 mới đánh giá trên tập kiểm tra cuối đã khóa.**

## 1. Những gì chương trình giữ cố định

- Mô hình: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Phiên bản chính xác: `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Điểm lấy đặc trưng: `model.model.visual`.
- Bộ mã hóa thị giác: FP32.
- Mô hình ngôn ngữ: FP16.
- Cơ chế chú ý: SDPA.
- Toàn bộ Qwen bị đóng băng.
- Mô-đun thích nghi chạy FP32 và là phần duy nhất được học.
- Không dùng LoRA, lượng tử hóa, OCR, Q-latent, dynamic gate hoặc phục hồi pixel.
- Không dùng ảnh đánh giá cuối để huấn luyện, chọn trọng số hay điều chỉnh siêu tham số.

Mô-đun thích nghi có kiến trúc:

```text
LayerNorm(2048)
→ Linear(2048, 256)
→ GELU
→ Dropout(0.0)
→ Linear(256, 2048)
→ cộng phần dư với đặc trưng đầu vào
```

Lớp `Linear` cuối được khởi tạo bằng 0. Mô-đun thích nghi của Giai đoạn 2 luôn được khởi tạo mới; trọng số chạy thử của Giai đoạn 1 không được dùng làm điểm bắt đầu.

## 2. Dữ liệu cần đặt trên máy H200

### Thầy cần tải những gì?

Để chạy Giai đoạn 2, thầy chỉ cần ba thành phần:

1. Kho mã nguồn Giai đoạn 2 này.
2. Dataset riêng tư chứa kết quả hoàn chỉnh của Giai đoạn 1: [textvqa_feature_restore_v1](https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1).
3. Mô hình `Qwen/Qwen2.5-VL-3B-Instruct`. Chương trình sẽ tải từ Hugging Face khi máy chủ có mạng; nếu máy chủ không có mạng thì nhóm phải gửi thêm bản mô hình cục bộ.

Dataset Giai đoạn 1 hiện có dung lượng khoảng 2,41 GB và chứa 4.422 tệp. Nó được tạo từ kết quả của sổ tay Kaggle trước đó tại:

```text
/kaggle/working/textvqa_feature_restore_v1.zip
```

Dataset đang ở chế độ riêng tư. Trước khi thầy tải, nhóm phải mở phần **Settings/Sharing** của Dataset và thêm đúng tài khoản Kaggle của thầy.

```text
https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1
```

**Không gửi nhầm bộ dữ liệu thử nghiệm nền.** Bộ [Data_suy_thoai (Text VQA)](https://www.kaggle.com/datasets/anhthu128/data-suy-thoai-text-vqa) và ảnh TextVQA gốc đã được dùng để tạo dữ liệu Giai đoạn 1. Chúng không phải đầu vào trực tiếp của chương trình Giai đoạn 2. Thầy cũng không cần tải lại bộ 1.000 ảnh đánh giá nền.

### Cách tải bằng trình duyệt

Sau khi được cấp quyền, thầy mở liên kết Dataset, bấm **Download**, chuyển tệp tải được lên máy chủ rồi chạy:

```bash
mkdir -p "$HOME/textvqa_phase2_download"
mkdir -p "$HOME/textvqa_phase2_data/phase1"
python3.11 -m zipfile -e \
  /duong/dan/da-tai/textvqa-feature-restore-v1.zip \
  "$HOME/textvqa_phase2_download"
cp -a \
  "$HOME/textvqa_phase2_download/phase1_dataset/textvqa_feature_restore_v1" \
  "$HOME/textvqa_phase2_data/phase1/"
```

### Cách tải thẳng trên máy chủ bằng Kaggle CLI

```bash
python3.11 -m venv "$HOME/.venvs/kaggle-cli"
"$HOME/.venvs/kaggle-cli/bin/python" -m pip install "kaggle==1.7.4.5"
"$HOME/.venvs/kaggle-cli/bin/kaggle" auth login --no-launch-browser

mkdir -p "$HOME/textvqa_phase2_download"
"$HOME/.venvs/kaggle-cli/bin/kaggle" datasets download \
  embechuangoan/textvqa-feature-restore-v1 \
  -p "$HOME/textvqa_phase2_download" \
  --unzip

mkdir -p "$HOME/textvqa_phase2_data/phase1"
cp -a \
  "$HOME/textvqa_phase2_download/phase1_dataset/textvqa_feature_restore_v1" \
  "$HOME/textvqa_phase2_data/phase1/"
```

Nếu Kaggle báo không có quyền truy cập, kiểm tra tài khoản vừa đăng nhập đã được nhóm thêm vào Dataset riêng tư hay chưa.

Sau khi giải nén, tệp sau phải tồn tại:

```bash
test -f "$HOME/textvqa_phase2_data/phase1/textvqa_feature_restore_v1/train_manifest.csv"
```

Nếu lệnh `test` không báo lỗi thì dùng:

```bash
export TEXTVQA_PHASE2_DATA="$HOME/textvqa_phase2_data"
export TEXTVQA_PHASE2_OUTPUT="$HOME/textvqa_phase2_output"
```

Kho mã không chứa ảnh, mô hình hoặc trọng số. Cấu trúc sau khi giải nén phải là:

```text
<DATA_ROOT>/
├── phase1/
│   └── textvqa_feature_restore_v1/
│       ├── split_summary.json
│       ├── benchmark_excluded_image_ids.csv
│       ├── train_manifest.csv
│       ├── dev_manifest.csv
│       ├── architecture_report.json
│       ├── injection_point_report.json
│       ├── adapter_config.json
│       ├── trainable_parameters.json
│       ├── smoke_summary.json
│       └── data/
│           ├── train/
│           │   ├── clean/<image_id>.jpg
│           │   └── degraded/<image_id>.png
│           └── dev/
│               ├── clean/<image_id>.jpg
│               └── degraded/<image_id>.png
└── models/
    └── Qwen2.5-VL-3B-Instruct/   # chỉ cần khi chạy không có mạng
```

Không cần đưa 1.000 ảnh đánh giá cuối lên máy cho Giai đoạn 2. Chương trình chỉ cần `benchmark_excluded_image_ids.csv` để chứng minh tập huấn luyện và phát triển không chứa các mã ảnh đó.

Dataset Giai đoạn 1 bắt buộc phải chứa tối thiểu:

- 2.000 ảnh sạch và 2.000 ảnh suy thoái thuộc tập huấn luyện.
- 200 ảnh sạch và 200 ảnh suy thoái thuộc tập phát triển.
- `train_manifest.csv` và `dev_manifest.csv`.
- `benchmark_excluded_image_ids.csv` chứa 1.000 mã ảnh bị loại.
- Các báo cáo kiến trúc và báo cáo chạy thử của Giai đoạn 1.

Không được chỉ gửi riêng các tệp CSV vì chương trình còn cần toàn bộ ảnh nằm trong thư mục `data/`.

Xem chi tiết tại [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).

## 3. Lấy mã và tạo môi trường riêng

Máy đích: Linux, Python 3.10 hoặc 3.11, CUDA và ít nhất một GPU NVIDIA H200 được hệ thống nhận diện.

```bash
git clone https://github.com/ducdanh2304/textvqa-feature-restore-phase2.git
cd textvqa-feature-restore-phase2
bash scripts/setup_env.sh
```

Lệnh trên tạo `.venv` ngay trong kho mã và không cài thư viện vào Python hệ thống.

Nếu máy chủ không có mạng, chuẩn bị trước thư mục gói cài đặt và bản mô hình cục bộ, rồi chạy:

```bash
bash scripts/setup_env.sh --offline --wheel-cache /duong/dan/toi/wheels
```

Bản mô hình cục bộ phải có `config.json`, đầy đủ trọng số, bộ tách từ và tệp `REVISION` chứa đúng:

```text
66285546d2b821cf421d4f5eb2576359d3770cd3
```

## 4. Kiểm tra không cần tải Qwen

Các phép kiểm tra trên CPU không tải mô hình và không cần GPU:

```bash
source .venv/bin/activate
pytest -q
```

Các phép kiểm tra xác nhận mô-đun thích nghi giữ nguyên kích thước tensor và không làm đổi đầu vào khi mới khởi tạo; mặt nạ bỏ qua token đệm; danh sách mẫu có mã trùng hoặc làm rò rỉ tập đánh giá cuối bị từ chối; dữ liệu đặc trưng cũ và trọng số sai nguồn gốc cũng bị từ chối.

## 5. Chạy Giai đoạn 2 trên H200

Có thể sửa hai đường dẫn đầu trong `configs/phase2_h200.yaml`, hoặc truyền đường dẫn trực tiếp trong mỗi lệnh.

### Bước 1 — kiểm tra dữ liệu, phần cứng và dung lượng đĩa

```bash
bash scripts/preflight.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Chỉ tiếp tục khi `preflight_report.json` có `status: READY`. Bước này chưa nạp mô hình.

### Bước 2 — trích và lưu đặc trưng

```bash
bash scripts/cache_features.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Qwen trích đúng 4.400 tensor: 2.000 cặp huấn luyện và 200 cặp phát triển, mỗi cặp gồm ảnh sạch và ảnh suy thoái. Mỗi tensor được kiểm tra ở FP32 trước khi lưu FP16. Phần dữ liệu lưu sẵn chứa mã kiểm tra, thống kê tensor, mã kiểm tra ảnh nguồn, cấu hình bộ xử lý, phiên bản mô hình và mã kiểm tra danh sách mẫu. Dữ liệu hợp lệ được dùng lại khi chạy tiếp; dữ liệu đã tồn tại nhưng sai nguồn gốc làm chương trình dừng.

### Bước 3 — huấn luyện mô-đun thích nghi từ đặc trưng đã lưu

```bash
bash scripts/train_adapter.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Cấu hình mặc định:

- Kích thước lô 16; chương trình kiểm tra bộ nhớ trước khi huấn luyện.
- AdamW, tốc độ học `1e-4`, hệ số suy giảm trọng số `0.01`.
- Giới hạn chuẩn gradient `1.0`.
- Tối đa 20 vòng huấn luyện.
- Dừng sớm sau 4 vòng liên tiếp mà tổng mất mát đặc trưng trên tập phát triển không cải thiện.
- Thứ tự dữ liệu huấn luyện ở vòng `e` được tạo bằng hạt ngẫu nhiên `42 + e`.
- Tập phát triển luôn dùng đúng 200 cặp theo thứ tự cố định.

Nếu cần chạy tiếp sau khi gián đoạn, phải chỉ rõ tệp trọng số Giai đoạn 2:

```bash
bash scripts/train_adapter.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT" \
  --resume "$TEXTVQA_PHASE2_OUTPUT/training/last_checkpoint.pt"
```

Chương trình từ chối trọng số chạy thử của Giai đoạn 1 hoặc trọng số có nguồn gốc không khớp.

### Bước 4 — kiểm tra trọng số tốt nhất

```bash
bash scripts/verify_checkpoint.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT" \
  --checkpoint "$TEXTVQA_PHASE2_OUTPUT/training/best_adapter.pt"
```

Bước này tạo một mô-đun thích nghi mới, nạp trọng số, đọc một cặp đặc trưng phát triển đã lưu và kiểm tra kích thước tensor, tính hữu hạn cùng hàm mất mát đặc trưng. Bước này không nạp Qwen và không sinh câu trả lời.

### Bước 5 — đóng gói kết quả

```bash
bash scripts/make_bundle.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Kết quả được đóng thành `$HOME/textvqa_phase2_output.zip`.

### Chạy toàn bộ bằng một lệnh

Chỉ dùng cho thư mục kết quả mới, chưa có trọng số huấn luyện:

```bash
bash scripts/run_all_phase2.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

## 6. Chạy bằng SLURM

Tệp [slurm/phase2_h200.sbatch](slurm/phase2_h200.sbatch) là mẫu. Cần thay `REPO_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT` và hai đường dẫn nhật ký trước khi chạy:

```bash
sbatch slurm/phase2_h200.sbatch
```

Kho mã không tự gửi tác vụ lên hệ thống hàng đợi.

## 7. Kết quả được tạo

```text
<OUTPUT_ROOT>/
├── preflight_report.json
├── preflight_report.md
├── environment.json
├── phase2_config_resolved.yaml
├── cache/
│   ├── manifest.jsonl
│   ├── cache_summary.json
│   └── features/*.pt
├── cache_report.json
├── cache_report.md
├── training/
│   ├── train_history.csv
│   ├── dev_history.csv
│   ├── best_adapter.pt
│   ├── last_checkpoint.pt
│   ├── emergency_checkpoint.pt       # chỉ có khi lỗi sau khi bắt đầu train
│   ├── checkpoint_manifest.json
│   ├── early_stopping.json
│   ├── training_summary.json
│   └── logs/
├── verification/
│   ├── checkpoint_verification.json
│   └── checkpoint_verification.md
├── figures/
└── artifact_manifest.json
```

## 8. Tệp cần giữ để sang Giai đoạn 3

- `training/best_adapter.pt`
- `phase2_config_resolved.yaml`
- `artifact_manifest.json`
- `training/checkpoint_manifest.json`
- `training/training_summary.json`
- `verification/checkpoint_verification.json`
- `cache/manifest.jsonl`
- Các danh sách mẫu Giai đoạn 1 và tệp loại trừ tập đánh giá cuối có mã kiểm tra trùng với tệp trọng số.

Các tensor đặc trưng đã lưu cần được giữ để kiểm toán và tái lập Giai đoạn 2. Khi chạy Giai đoạn 3, mô-đun thích nghi mới được gắn vào Qwen và đánh giá trên tập kiểm tra cuối bằng một giao thức riêng.

## 9. Giới hạn hiện tại

Các phép kiểm tra trên CPU đã xác nhận logic cục bộ. Những phần sau chỉ được xác nhận khi chạy trên máy chủ thật:

- Bản Qwen đúng phiên bản có tải và nạp được hay không.
- 4.400 lượt trích đặc trưng trên H200.
- Mức dùng VRAM cao nhất, dung lượng đặc trưng lưu sẵn và thời gian thực tế.
- Kích thước lô 16 có vượt qua kiểm tra bộ nhớ hay không.
- Quá trình tối đa 20 vòng huấn luyện và cơ chế dừng sớm.
- Giới hạn thời gian cụ thể của hệ thống SLURM.

Không được dùng mất mát đặc trưng của Giai đoạn 2 để tuyên bố mô hình đã tăng độ chính xác TextVQA.
