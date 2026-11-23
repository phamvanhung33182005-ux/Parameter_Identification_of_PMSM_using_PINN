"""Mạng MLP cơ sở: ánh xạ (ud, uq, id, iq, nm) sang (Rs, Ld, Lq), chỉ học từ nhãn."""  # Mô tả mô-đun
import torch  # Thư viện học sâu
import torch.nn as nn  # Các lớp mạng nơ-ron

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}  # Bảng hàm kích hoạt


class BaselineMLP(nn.Module):  # Lớp mạng MLP cơ sở
    """MLP với chuẩn hóa đầu vào cố định và đầu ra bị chặn trong khoảng vật lý."""  # Mô tả lớp

    def __init__(self, in_scale, out_lo, out_hi, hidden_sizes=(32, 32), activation="tanh"):  # Hàm khởi tạo
        """Dựng mạng với hệ số chuẩn hóa đầu vào và cận đầu ra."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.hidden_sizes, self.activation = list(hidden_sizes), activation  # Lưu kiến trúc
        self.register_buffer("in_scale", torch.tensor(in_scale, dtype=torch.float32))  # Hệ số chia đầu vào
        self.register_buffer("out_lo", torch.tensor(out_lo, dtype=torch.float32))  # Cận dưới đầu ra
        self.register_buffer("out_hi", torch.tensor(out_hi, dtype=torch.float32))  # Cận trên đầu ra
        sizes = [len(in_scale)] + self.hidden_sizes + [len(out_lo)]  # Danh sách kích thước các lớp
        layers = []  # Danh sách lớp
        for i in range(len(sizes) - 1):  # Duyệt từng cặp lớp liên tiếp
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))  # Thêm lớp tuyến tính
            if i < len(sizes) - 2:  # Nếu chưa phải lớp cuối
                layers.append(ACTIVATIONS[activation]())  # Thêm hàm kích hoạt
        self.net = nn.Sequential(*layers)  # Gộp thành mạng tuần tự
        self.init_weights()  # Khởi tạo trọng số

    def init_weights(self):  # Khởi tạo trọng số
        """Khởi tạo Xavier cho trọng số và bằng 0 cho độ lệch."""  # Mô tả hàm
        for m in self.net:  # Duyệt từng lớp
            if isinstance(m, nn.Linear):  # Chỉ xử lý lớp tuyến tính
                nn.init.xavier_uniform_(m.weight)  # Khởi tạo Xavier
                nn.init.zeros_(m.bias)  # Độ lệch bằng 0

    def forward(self, x):  # Lan truyền thuận
        """Nhận đầu vào vật lý (N,5), trả về (Rs, Ld, Lq) theo đơn vị SI (N,3)."""  # Mô tả hàm
        z = self.net(x / self.in_scale)  # Chuẩn hóa rồi qua mạng
        return self.out_lo + (self.out_hi - self.out_lo) * torch.sigmoid(z)  # Ánh xạ vào khoảng [lo, hi]

    def count_parameters(self):  # Đếm tham số
        """Trả về tổng số tham số huấn luyện được."""  # Mô tả hàm
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # Cộng số phần tử

    def get_config(self):  # Lấy cấu hình
        """Trả về từ điển đủ để dựng lại mạng."""  # Mô tả hàm
        return {"in_scale": self.in_scale.tolist(), "out_lo": self.out_lo.tolist(), "out_hi": self.out_hi.tolist(), "hidden_sizes": self.hidden_sizes, "activation": self.activation}  # Gói cấu hình


def build_baseline(cfg, stats):  # Hàm dựng mạng từ cấu hình
    """Dựng BaselineMLP từ cấu hình và thống kê nhãn."""  # Mô tả hàm
    m = cfg["motor"]  # Nhóm thông số động cơ
    u_base = m["u_dc"] / 3 ** 0.5  # Điện áp cơ sở (V)
    in_scale = [u_base, u_base, m["i_rated_peak"], m["i_rated_peak"], m["n_max"]]  # Hệ số chuẩn hóa đầu vào
    lo = [stats[k]["lo"] for k in ("Rs", "Ld", "Lq")]  # Cận dưới đầu ra
    hi = [stats[k]["hi"] for k in ("Rs", "Ld", "Lq")]  # Cận trên đầu ra
    return BaselineMLP(in_scale, lo, hi, cfg["model"]["hidden_sizes"], cfg["model"]["activation"])  # Dựng mạng
