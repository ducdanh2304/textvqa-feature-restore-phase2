# Cấu trúc dữ liệu bắt buộc

Truyền thư mục cha bằng đối số `--data-root`, hoặc khai báo `paths.data_root` trong tệp YAML:

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
    └── Qwen2.5-VL-3B-Instruct/       # bản mô hình cục bộ, chỉ cần khi chạy không có mạng
        ├── config.json
        ├── REVISION                   # mã cam kết chính xác đã khóa
        └── ...
```

Chương trình kiểm tra dùng các đường dẫn chuẩn dạng `data/<split>/<condition>/<image_id>.<ext>`. Đường dẫn ghi trong các tệp CSV của Giai đoạn 1 có thể thuộc máy cũ, nên không được dùng để thay thế tệp trên máy hiện tại. Mỗi tệp ở đường dẫn chuẩn phải tồn tại, có tên không kèm phần mở rộng trùng với `image_id`, khớp mã SHA256 đã ghi, mở được và có kích thước giống ảnh còn lại trong cùng cặp.

Khi chạy không có mạng, tệp `REVISION` phải chứa đúng nội dung sau:

```text
66285546d2b821cf421d4f5eb2576359d3770cd3
```

Không cần đưa 1.000 ảnh đánh giá cuối lên máy chủ trong Giai đoạn 2. Chỉ cần tệp chứa danh sách mã ảnh bị loại để chứng minh rằng dữ liệu huấn luyện và phát triển không trùng với tập đánh giá cuối.

