# 🛡️ Hệ Thống Dự Báo Rủi Ro Tín Dụng & Phát Hiện Gian Lận Web App

Ứng dụng web được xây dựng dựa trên nền tảng **Streamlit** kết hợp mô hình học máy chuyên sâu giúp tự động hóa quá trình phân tích dữ liệu giao dịch tài chính, huấn luyện thuật toán thông minh và đưa ra cảnh báo sớm các tài khoản có nguy cơ rủi ro cao hoặc vỡ nợ dựa trên quy trình nghiên cứu tại file notebook huấn luyện gốc.

## 🗂️ Cấu trúc Nghiệp vụ & Mô hình hóa
Dựa trên kết quả thực nghiệm trong notebook, ứng dụng tích hợp đồng bộ 3 họ thuật toán phân loại có giám sát ưu việt:
1. **Random Forest Classifier**: Mô hình cho thấy hiệu năng vượt trội với độ chính xác tổng thể đạt **95%**, khả năng giảm thiểu cảnh báo giả lý tưởng và độ nhạy bắt vết rủi ro tối ưu nhất.
2. **Logistic Regression (Hồi quy Logistic)**.
3. **Decision Tree (Cây quyết định)**.

Tập biến đặc trưng đầu vào hệ thống trích xuất bao gồm **14 chỉ số biến số liên tục cấu trúc mã hóa (`X_1` đến `X_14`)** để đưa ra dự báo phân loại nhãn mục tiêu nhị phân **`default`** (0: Hồ sơ tài khoản an toàn | 1: Hồ sơ phát hiện nguy cơ cao/gian lận vỡ nợ).

---

## 🛠️ Hướng dẫn cài đặt và khởi chạy hệ thống

### Bước 1: Khởi tạo không gian môi trường và tải thư viện liên quan
Mở terminal/command prompt tại thư mục chứa các file ứng dụng và thực hiện cài đặt các thư viện cần thiết thông qua file cấu hình đi kèm:
```bash
pip install -r requirements.txt
