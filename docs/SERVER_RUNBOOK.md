# Hướng dẫn chạy trên máy chủ dành cho giảng viên

1. Tải kho mã nguồn.
2. Yêu cầu nhóm cấp quyền vào Dataset riêng tư [textvqa_feature_restore_v1](https://www.kaggle.com/datasets/embechuangoan/textvqa-feature-restore-v1). Đây là kết quả của sổ tay Kaggle Giai đoạn 1, không phải bộ 1.000 ảnh thử nghiệm nền.
3. Tải Dataset, giải nén và đặt lại đúng cấu trúc:

   ```bash
   mkdir -p "$HOME/textvqa_phase2_download"
   python3.11 -m zipfile -e \
     /duong/dan/da-tai/textvqa-feature-restore-v1.zip \
     "$HOME/textvqa_phase2_download"
   mkdir -p "$HOME/textvqa_phase2_data/phase1"
   cp -a \
     "$HOME/textvqa_phase2_download/phase1_dataset/textvqa_feature_restore_v1" \
     "$HOME/textvqa_phase2_data/phase1/"
   test -f "$HOME/textvqa_phase2_data/phase1/textvqa_feature_restore_v1/train_manifest.csv"
   export TEXTVQA_PHASE2_DATA="$HOME/textvqa_phase2_data"
   export TEXTVQA_PHASE2_OUTPUT="$HOME/textvqa_phase2_output"
   ```

4. Đối chiếu thư mục vừa giải nén với [cấu trúc dữ liệu](DATA_LAYOUT.md).
5. Tạo một bản sao của `configs/phase2_h200.yaml` để chỉnh, hoặc dùng hai biến đường dẫn ở trên.
6. Tạo môi trường riêng:

   ```bash
   bash scripts/setup_env.sh
   ```

7. Kiểm tra dữ liệu, cấu hình và phần cứng. Đọc cả hai báo cáo và dừng nếu trạng thái không phải `READY`:

   ```bash
   bash scripts/preflight.sh --config configs/phase2_h200.yaml \
     --data-root "$TEXTVQA_PHASE2_DATA" --output-root "$TEXTVQA_PHASE2_OUTPUT"
   ```

8. Trích và lưu đúng 4.400 tensor đặc trưng thị giác đã đóng băng:

   ```bash
   bash scripts/cache_features.sh --config configs/phase2_h200.yaml \
     --data-root "$TEXTVQA_PHASE2_DATA" --output-root "$TEXTVQA_PHASE2_OUTPUT"
   ```

9. Huấn luyện mới mô-đun thích nghi có kết nối phần dư:

   ```bash
   bash scripts/train_adapter.sh --config configs/phase2_h200.yaml \
     --data-root "$TEXTVQA_PHASE2_DATA" --output-root "$TEXTVQA_PHASE2_OUTPUT"
   ```

10. Kiểm tra trọng số tốt nhất và đóng gói kết quả:

   ```bash
   bash scripts/verify_checkpoint.sh --config configs/phase2_h200.yaml \
     --data-root "$TEXTVQA_PHASE2_DATA" --output-root "$TEXTVQA_PHASE2_OUTPUT" \
     --checkpoint "$TEXTVQA_PHASE2_OUTPUT/training/best_adapter.pt"
   bash scripts/make_bundle.sh --config configs/phase2_h200.yaml \
     --data-root "$TEXTVQA_PHASE2_DATA" --output-root "$TEXTVQA_PHASE2_OUTPUT"
   ```

Nếu cần tiếp tục một lượt chạy bị gián đoạn, phải chỉ rõ tệp trọng số của Giai đoạn 2:

```bash
bash scripts/train_adapter.sh --config configs/phase2_h200.yaml \
  --data-root /absolute/data --output-root /absolute/output \
  --resume /absolute/output/training/last_checkpoint.pt
```

Không tiếp tục huấn luyện từ trọng số chạy thử của Giai đoạn 1. Giai đoạn 2 không sinh câu trả lời ngôn ngữ và không tính độ chính xác TextVQA.

