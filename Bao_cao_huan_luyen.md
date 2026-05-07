# BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH

**Tên:** Trần Gia Khánh
**MSV:** 2A202600293
**Ngày báo cáo:** 07/05/2026

---

## 1. Bộ siêu tham số đã chọn và lý do
Dựa trên chuỗi các thử nghiệm (Step 1), mô hình cuối cùng được lựa chọn đạt hiệu suất cao nhất với các thông số sau:

- **Mô hình chính:** HistGradientBoostingClassifier (được huấn luyện trên dữ liệu tổng hợp).
- **Bộ siêu tham số tối ưu:**
  - `max_iter`: 1000
  - `max_depth`: 10
  - `learning_rate`: 0.03
  - `min_samples_leaf`: 10
- **Kết quả đạt được:** **Accuracy: 0.7500** | **F1-Score: 0.7495**.

**Lý do lựa chọn:**
- **Vượt ngưỡng hiệu suất:** Các thử nghiệm ban đầu với Random Forest và Extra Trees chỉ đạt mức Accuracy tối đa khoảng 0.6780, chưa đạt kỳ vọng cao hơn.
- **Tận dụng dữ liệu mới:** Sau khi tích hợp dữ liệu Phase 2 vào Phase 1 (tăng từ 2000 lên 3000 mẫu), mô hình HistGradientBoosting đã thể hiện khả năng học vượt trội, tận dụng tốt lượng dữ liệu bổ sung để cải thiện độ chính xác từ 0.6460 lên 0.7500.
- **Tính ổn định:** Bộ tham số này giúp cân bằng giữa độ phức tạp (depth 10) và tốc độ hội tụ (learning rate 0.03), tránh hiện tượng overfitting thường gặp khi để depth tự do.

---

## 2. Khó khăn gặp phải và cách giải quyết

### Khó khăn 1: Hiệu suất mô hình bị "bão hòa" ở mức thấp
- **Mô tả:** Trong các lần chạy đầu tiên với Random Forest, dù đã tăng `n_estimators` lên đến 800, Accuracy vẫn chỉ dao động quanh mức 0.67 - 0.68. Việc điều chỉnh các tham số nhỏ lẻ không mang lại sự đột biến.
- **Giải quyết:** Nhận định rằng vấn đề nằm ở số lượng dữ liệu huấn luyện. Giải pháp là thực hiện quy trình **Data Augmentation** (chạy script `add_new_data.py`) để bổ sung 1000 mẫu từ Phase 2. Kết quả là hiệu suất tăng vọt ~10% ngay sau đó.

### Khó khăn 2: Mô hình bị Underfitting do giới hạn tham số quá chặt
- **Mô tả:** Khi thử nghiệm với `max_depth: 3` và `n_estimators: 50`, Accuracy giảm xuống còn 0.5580. Điều này cho thấy mô hình quá đơn giản để bắt được các đặc trưng của dữ liệu.
- **Giải quyết:** Nới lỏng các ràng buộc, cho phép mô hình phát triển sâu hơn (depth 10-20 hoặc null) và sử dụng số lượng cây lớn hơn để đảm bảo khả năng học tập tối ưu.

### Khó khăn 3: Cảnh báo bảo mật và phiên bản MLflow
- **Mô tả:** Hệ thống liên tục đưa ra cảnh báo về việc sử dụng `pickle` để lưu trữ mô hình và khuyến nghị dùng `skops`. Đồng thời cảnh báo về việc `artifact_path` bị thay thế bởi `name`.
- **Giải quyết:** Đã ghi nhận các cảnh báo này để cập nhật mã nguồn trong các phiên bản tiếp theo (Refactoring), đảm bảo hệ thống tuân thủ các tiêu chuẩn bảo mật và API mới nhất của MLflow.

---
**Người thực hiện:** Antigravity AI Assistant
