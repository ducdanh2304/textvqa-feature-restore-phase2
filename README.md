# Phase 2: Huấn luyện mô-đun phục hồi đặc trưng cho ảnh Scene Text suy thoái

Kho mã này huấn luyện một mô-đun nhỏ để biến **đặc trưng của ảnh suy thoái** trở nên gần với **đặc trưng của ảnh sạch tương ứng**. Quá trình phục hồi diễn ra trên đặc trưng thị giác (latent feature), không tạo lại ảnh ở mức pixel.

## 1. Phase 2 làm gì?

```text
Ảnh sạch ──────► Vision Encoder Qwen ─► đặc trưng sạch ───────────────┐
                                                                    │ so sánh
Ảnh suy thoái ─► Vision Encoder Qwen ─► đặc trưng suy thoái ─► Adapter
                                                                    │
                                                                    ▼
                                                        đặc trưng được phục hồi
```

Trong quá trình huấn luyện:

1. Đóng băng toàn bộ `Qwen/Qwen2.5-VL-3B-Instruct`.
2. Trích đặc trưng từ từng cặp ảnh sạch và ảnh suy thoái.
3. Đưa đặc trưng suy thoái qua adapter.
4. So sánh kết quả với đặc trưng sạch.
5. Chỉ cập nhật trọng số của adapter.
6. Chọn checkpoint tốt nhất dựa trên tập phát triển.

Adapter có kiến trúc:

```text
LayerNorm(2048)
→ Linear(2048, 256)
→ GELU
→ Linear(256, 2048)
→ cộng phần dư với đặc trưng đầu vào
```

### Đầu vào

- 2.000 cặp ảnh sạch và ảnh suy thoái để huấn luyện.
- 200 cặp ảnh sạch và ảnh suy thoái để chọn checkpoint.
- Các manifest và báo cáo nguồn gốc dữ liệu từ Phase 1.
- Mô hình `Qwen/Qwen2.5-VL-3B-Instruct`.

### Đầu ra chính

- `best_adapter.pt`: trọng số adapter tốt nhất.
- Lịch sử train/dev loss.
- Báo cáo kiểm tra checkpoint và nguồn gốc thí nghiệm.
- Gói `textvqa_phase2_output.zip` để dùng ở Phase 3.

> Phase 2 chỉ huấn luyện khả năng phục hồi đặc trưng. Nó chưa sinh câu trả lời và chưa tính TextVQA Accuracy. Việc gắn adapter vào MLLM và đánh giá VQA thuộc Phase 3.

## 2. Yêu cầu hệ thống

- Linux.
- Python 3.10 hoặc 3.11.
- CUDA và GPU NVIDIA; cấu hình mặc định dành cho H200.
- Có mạng để tải Qwen từ Hugging Face, hoặc có sẵn mô hình cục bộ.
- Khoảng trống đĩa đủ cho dataset, mô hình và 4.400 tensor đặc trưng.

## 3. Chuẩn bị dữ liệu Phase 1

Dataset cần dùng:

[textvqa-feature-restore-v1 trên Kaggle](https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1)

Dataset hiện ở chế độ riêng tư. Tài khoản Kaggle dùng để tải phải được chủ sở hữu cấp quyền trước.

Không cần tải lại bộ TextVQA gốc, bộ ảnh suy thoái ban đầu hoặc 1.000 ảnh benchmark. Phase 2 chỉ sử dụng dataset kết quả của Phase 1 ở liên kết trên.

### Cách A: tải bằng trình duyệt

1. Mở liên kết dataset.
2. Bấm **Download**.
3. Chuyển tệp ZIP lên máy chạy.
4. Giải nén và đưa dữ liệu về đúng vị trí:

```bash
mkdir -p "$HOME/textvqa_phase2_download"
mkdir -p "$HOME/textvqa_phase2_data/phase1"

python3.11 -m zipfile -e \
  /duong/dan/toi/textvqa-feature-restore-v1.zip \
  "$HOME/textvqa_phase2_download"

cp -a \
  "$HOME/textvqa_phase2_download/phase1_dataset/textvqa_feature_restore_v1" \
  "$HOME/textvqa_phase2_data/phase1/"
```

### Cách B: tải bằng Kaggle CLI

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

### Kiểm tra dữ liệu

```bash
test -f "$HOME/textvqa_phase2_data/phase1/textvqa_feature_restore_v1/train_manifest.csv" \
  && echo "Dataset hợp lệ"
```

Cấu trúc tối thiểu:

```text
textvqa_phase2_data/
└── phase1/
    └── textvqa_feature_restore_v1/
        ├── train_manifest.csv
        ├── dev_manifest.csv
        ├── benchmark_excluded_image_ids.csv
        ├── split_summary.json
        ├── architecture_report.json
        ├── injection_point_report.json
        ├── adapter_config.json
        └── data/
            ├── train/
            │   ├── clean/
            │   └── degraded/
            └── dev/
                ├── clean/
                └── degraded/
```

Không chỉ sao chép các tệp CSV. Chương trình cần toàn bộ ảnh trong thư mục `data/`.

## 4. Tải mã nguồn và tạo môi trường

```bash
git clone https://github.com/ducdanh2304/textvqa-feature-restore-phase2.git
cd textvqa-feature-restore-phase2
bash scripts/setup_env.sh
source .venv/bin/activate
```

Thiết lập đường dẫn dùng chung:

```bash
export TEXTVQA_PHASE2_DATA="$HOME/textvqa_phase2_data"
export TEXTVQA_PHASE2_OUTPUT="$HOME/textvqa_phase2_output"
```

Nếu máy không có mạng, chuẩn bị wheel cache và bản Qwen cục bộ rồi chạy:

```bash
bash scripts/setup_env.sh --offline --wheel-cache /duong/dan/toi/wheels
```

Mô hình cục bộ phải có đầy đủ cấu hình, trọng số, tokenizer và tệp `REVISION` chứa:

```text
66285546d2b821cf421d4f5eb2576359d3770cd3
```

## 5. Kiểm tra mã nguồn trên CPU

Bước này không tải Qwen và không cần GPU:

```bash
pytest -q
```

## 6. Chạy Phase 2 từng bước

### Bước 1: kiểm tra dữ liệu và máy chạy

```bash
bash scripts/preflight.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Chỉ tiếp tục khi `$TEXTVQA_PHASE2_OUTPUT/preflight_report.json` có `status: READY`.

### Bước 2: trích đặc trưng bằng Qwen

```bash
bash scripts/cache_features.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Bước này trích và lưu 4.400 tensor: 2.000 cặp train và 200 cặp dev, mỗi cặp gồm ảnh sạch và ảnh suy thoái. Đây thường là bước tốn thời gian nhất.

### Bước 3: huấn luyện adapter

```bash
bash scripts/train_adapter.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Cấu hình mặc định:

- AdamW, learning rate `1e-4`.
- Tối đa 20 epoch.
- Early stopping sau 4 epoch liên tiếp không cải thiện dev loss.
- Chỉ adapter được cập nhật; Qwen luôn đóng băng.

Nếu quá trình bị gián đoạn:

```bash
bash scripts/train_adapter.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT" \
  --resume "$TEXTVQA_PHASE2_OUTPUT/training/last_checkpoint.pt"
```

### Bước 4: kiểm tra checkpoint tốt nhất

```bash
bash scripts/verify_checkpoint.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT" \
  --checkpoint "$TEXTVQA_PHASE2_OUTPUT/training/best_adapter.pt"
```

Bước này kiểm tra checkpoint có đúng kiến trúc, đúng nguồn gốc, không chứa giá trị lỗi và có thể phục hồi đặc trưng dev hay không.

### Bước 5: đóng gói kết quả

```bash
bash scripts/make_bundle.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Tệp cuối:

```text
$HOME/textvqa_phase2_output.zip
```

## 7. Chạy toàn bộ tự động

Sau khi chuẩn bị dữ liệu và môi trường:

```bash
bash scripts/run_all_phase2.sh \
  --config configs/phase2_h200.yaml \
  --data-root "$TEXTVQA_PHASE2_DATA" \
  --output-root "$TEXTVQA_PHASE2_OUTPUT"
```

Chỉ dùng lệnh này với thư mục output mới. Khi tiếp tục một phiên train bị gián đoạn, dùng `--resume` ở Bước 3.

## 8. Kết quả sau khi chạy

```text
textvqa_phase2_output/
├── preflight_report.json
├── environment.json
├── phase2_config_resolved.yaml
├── cache/
│   ├── manifest.jsonl
│   ├── cache_summary.json
│   └── features/*.pt
├── training/
│   ├── train_history.csv
│   ├── dev_history.csv
│   ├── best_adapter.pt
│   ├── last_checkpoint.pt
│   ├── checkpoint_manifest.json
│   ├── early_stopping.json
│   └── training_summary.json
├── verification/
│   └── checkpoint_verification.json
└── artifact_manifest.json
```

Các tệp quan trọng cần giữ:

- `training/best_adapter.pt`
- `phase2_config_resolved.yaml`
- `artifact_manifest.json`
- `training/checkpoint_manifest.json`
- `training/training_summary.json`
- `verification/checkpoint_verification.json`

## 9. Chạy qua SLURM

Sửa `REPO_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT` và đường dẫn log trong `slurm/phase2_h200.sbatch`, sau đó chạy:

```bash
sbatch slurm/phase2_h200.sbatch
```

## 10. Giới hạn của Phase 2

- Chưa đánh giá TextVQA Accuracy hoặc ANLS.
- Chưa chứng minh adapter giúp MLLM trả lời tốt hơn.
- Chưa xử lý theo câu hỏi hoặc tập trung riêng vào vùng chữ.
- Kết quả phải được đánh giá trên benchmark đã khóa ở Phase 3.

Không dùng việc dev feature loss giảm để kết luận độ chính xác VQA đã tăng.

Xem thêm cấu trúc dữ liệu tại [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).
