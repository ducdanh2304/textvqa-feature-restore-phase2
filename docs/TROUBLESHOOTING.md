# Xử lý sự cố

## Thiếu mô hình hoặc gặp lỗi khi chạy không có mạng

Dùng `configs/phase2_h200_offline.yaml`, truyền thêm `--model-path`, rồi kiểm tra thư mục mô hình có `config.json`, đầy đủ trọng số, bộ tách từ và tệp `REVISION` chứa mã cam kết đã khóa. Tạo môi trường bằng `bash scripts/setup_env.sh --offline --wheel-cache /path/to/wheels`.

## Dữ liệu đặc trưng lưu sẵn không khớp hoặc đã cũ

Chương trình chủ động từ chối dữ liệu lưu sẵn nếu nguồn gốc của mô hình, bộ xử lý, ảnh, danh sách mẫu, kiểu số hoặc mã triển khai khác với cấu hình hiện tại. Không sửa siêu dữ liệu bằng tay. Hãy đổi tên thư mục kết quả cũ, rồi chạy lại bước kiểm tra và trích đặc trưng vào một thư mục kết quả trống.

## Sai phiên bản mô hình

Khi có mạng, chương trình luôn yêu cầu mã cam kết `66285546d2b821cf421d4f5eb2576359d3770cd3`. Khi không có mạng, mã này phải xuất hiện trong đường dẫn của bản mô hình hoặc trong tệp `REVISION`. Tên nhánh không đủ để xác nhận đúng phiên bản.

## Hết bộ nhớ GPU khi trích đặc trưng

Mặc định bước trích đặc trưng xử lý một mẫu mỗi lần. Kiểm tra chỉ có một tiến trình dùng H200. Không tự ý giảm kiểu số hoặc giới hạn điểm ảnh. Dùng `nvidia-smi`, kết thúc các tác vụ không liên quan rồi chạy lại; các tệp đã ghi hoàn chỉnh và hợp lệ sẽ được tái sử dụng.

## Hết bộ nhớ GPU khi dò kích thước lô huấn luyện

Nếu kích thước lô được đặt cố định nhưng không đủ bộ nhớ, chương trình sẽ dừng thay vì âm thầm giảm giá trị. Hãy đổi `training.batch_size` thành 8, 16 hoặc 32 trong một cấu hình thí nghiệm mới. Nếu đặt `auto`, giá trị đã thử thành công sẽ được ghi vào nhật ký và tệp trọng số.

## Kiểu số không khớp

Cấu hình mặc định duy nhất được chấp nhận là: bộ mã hóa thị giác FP32, mô hình ngôn ngữ FP16, mô-đun thích nghi FP32 và đặc trưng lưu sẵn FP16. Không che lỗi bằng cách ép kiểu âm thầm; hãy kiểm tra phiên bản mô hình và siêu dữ liệu Giai đoạn 1.

## Dữ liệu đặc trưng lưu sẵn bị hỏng

Mỗi tệp tensor có mã kiểm tra nội dung tensor và mã kiểm tra tệp trong danh sách dữ liệu đặc trưng. Nếu phát hiện hỏng, chương trình phải dừng. Sau khi giữ lại nhật ký để đối chiếu, xóa thủ công đúng tệp không hợp lệ rồi chạy lại `cache-features`; chương trình sẽ tạo lại bản ghi còn thiếu theo cách ghi nguyên tử.

## Thiếu tệp của Giai đoạn 1

Dùng đúng cây thư mục trong `docs/DATA_LAYOUT.md`. Phải có đầy đủ hai thư mục ảnh sạch và ảnh suy thoái theo cặp. Tệp ZIP chưa giải nén không phải đầu vào hợp lệ.

## Không thể tiếp tục vì nguồn gốc thí nghiệm không khớp

Việc tiếp tục yêu cầu cùng danh sách mẫu, danh sách đặc trưng lưu sẵn, cấu hình đã phân giải, kích thước lô, kiểu số, phiên bản mã triển khai, định danh mô hình và phiên bản mô hình. Nếu chủ động thay đổi bất kỳ yếu tố nào, hãy dùng một thư mục kết quả mới. Chương trình từ chối trọng số chạy thử của Giai đoạn 1.

