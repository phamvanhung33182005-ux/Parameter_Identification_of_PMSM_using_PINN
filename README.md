# Nhận dạng tham số động cơ PMSM: MLP nhận dạng Ld, Lq và bộ ước lượng Rs đại số

Đồ án tốt nghiệp cử nhân, ngành Kỹ thuật điều khiển và Tự động hóa, Đại học Bách khoa Hà Nội.

## Mục tiêu
Nhận dạng điện cảm Ld, Lq và điện trở stator Rs của động cơ PMSM từ ud, uq, id, iq, nm.
Huấn luyện ngoại tuyến; vận hành trực tuyến chỉ thực hiện lan truyền thuận của MLP với trọng số đã đóng băng và bộ ước lượng Rs đại số.

## Phương pháp
1. Mạng MLP ánh xạ (id, iq) sang (Ld, Lq), huấn luyện có nhãn bằng sai số bình phương tương đối; đầu ra bị chặn trong khoảng nhãn của tập huấn luyện.
2. Rs không do mạng dự đoán. Rs được tính đại số từ phương trình trục q với Ld, Lq của MLP rồi lọc tỷ số các tổng (dạng `ratio` hoặc `integral`, cửa sổ `rect` hoặc `ema`),
   với ba lớp giới hạn: loại mẫu thô ngoài dải vật lý [0,9·Rs(20 °C); 1,1·Rs(128 °C)], số mẫu hợp lệ tối thiểu mỗi cửa sổ, cắt đầu ra về dải vật lý.
3. Từ thông nam châm Psi_m = 0,23191792 Wb là hằng số cài đặt của mô phỏng: cố định, không hiệu chỉnh, phải giữ đủ 8 chữ số có nghĩa.
4. Đối chứng: Rs tính bằng cùng bộ ước lượng nhưng dùng nhãn Ld, Lq, để tách sai số do MLP khỏi sai số do bộ ước lượng và dữ liệu.

## Cấu trúc kho
- `configs/config.yaml`: cấu hình trung tâm.
- `src/data/`: tiền xử lý, lớp Dataset, DataLoader.
- `src/models/`: ld_lq_mlp.py.
- `src/training/`: trainer.py (hàm mất mát và bộ huấn luyện).
- `src/evaluation/`: metrics.py (chỉ số và bộ ước lượng Rs), test.py.
- `src/deployment/`: xuất mã C cho DSP (sẽ cập nhật theo mô hình mới).
- `src/utils/`: tiện ích dùng chung.
- `notebooks/`: sáu sổ tay Colab chạy độc lập.

## Thứ tự chạy sổ tay
01 thiết lập, 02 tiền xử lý, 03 Dataset/DataLoader, 04 MLP Ld, Lq, 05 huấn luyện, 06 đánh giá (Ld, Lq, Rs).

## Chuẩn bị trên Colab
Lưu mã truy cập GitHub vào Colab Secrets với khóa `GITHUB_TOKEN` và bật quyền truy cập cho sổ tay.
Dữ liệu thô đặt tại `dataset/raw` trên Google Drive; trọng số và kết quả lưu trên Drive, không đưa lên GitHub.
