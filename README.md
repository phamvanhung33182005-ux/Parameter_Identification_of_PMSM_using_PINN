# Nhận dạng tham số động cơ PMSM bằng mạng nơ-ron tuân theo quy luật vật lý (PINN)

Đồ án tốt nghiệp cử nhân, ngành Kỹ thuật điều khiển và Tự động hóa, Đại học Bách khoa Hà Nội.

## Mục tiêu
Nhận dạng điện trở stator Rs và các điện cảm Ld, Lq của động cơ PMSM từ ud, uq, id, iq, nm.
Huấn luyện ngoại tuyến; vận hành trực tuyến chỉ thực hiện lan truyền thuận với trọng số đã đóng băng.

Phương án lai (P4): PINN nhận dạng Ld và Lq; Rs tính đại số từ phương trình trục q rồi lọc tỷ số các tổng (dạng `ratio` hoặc `integral`, cửa sổ `rect` hoặc `ema`),
với ba lớp giới hạn: loại mẫu thô ngoài dải vật lý [0,9·Rs(20 °C); 1,1·Rs(128 °C)], số mẫu hợp lệ tối thiểu mỗi cửa sổ, cắt đầu ra về dải vật lý.
Từ thông nam châm Psi_m = 0,23191792 Wb là hằng số cài đặt của mô phỏng: cố định, không hiệu chỉnh, phải giữ đủ 8 chữ số có nghĩa.

## Cấu trúc kho
- `configs/config.yaml`: cấu hình trung tâm.
- `src/data/`: tiền xử lý, lớp Dataset, DataLoader.
- `src/models/`: baselineMLP.py, pinn_model.py.
- `src/training/`: loss_functions.py, trainer.py.
- `src/evaluation/`: metrics.py, test.py, hil_analysis.py.
- `src/deployment/`: xuất mã C cho DSP (mạng PINN và bộ ước lượng Rs).
- `src/utils/`: tiện ích dùng chung.
- `notebooks/`: mười sổ tay Colab chạy độc lập.

## Thứ tự chạy sổ tay
01 thiết lập, 02 tiền xử lý, 03 Dataset/DataLoader, 04 MLP cơ sở, 05 PINN, 06 hàm mất mát,
07 huấn luyện, 08 đánh giá, 09 xuất mã C, 10 phân tích HIL và thực nghiệm.

## Chuẩn bị trên Colab
Lưu mã truy cập GitHub vào Colab Secrets với khóa `GITHUB_TOKEN` và bật quyền truy cập cho sổ tay.
Dữ liệu thô đặt tại `dataset/raw` trên Google Drive; trọng số và kết quả lưu trên Drive, không đưa lên GitHub.
