"""Mô hình PINN: cùng kiến trúc với MLP cơ sở, bổ sung phương trình điện áp d-q."""  # Mô tả mô-đun
import torch  # Thư viện học sâu
from src.models.baselineMLP import BaselineMLP  # Lớp mạng cơ sở


def pmsm_voltage_residual(params, aux, phi_m):  # Hàm sai số phương trình điện áp
    """Tính sai số (V) của hai phương trình điện áp d-q với tham số dự đoán."""  # Mô tả hàm
    rs, ld, lq = params[:, 0], params[:, 1], params[:, 2]  # Tách ba tham số
    ud, uq, i_d, i_q, w, did, diq = [aux[:, k] for k in range(7)]  # Tách bảy đại lượng phụ
    res_d = ud - (rs * i_d + ld * did - w * lq * i_q)  # Sai số phương trình trục d
    res_q = uq - (rs * i_q + lq * diq + w * ld * i_d + w * phi_m)  # Sai số phương trình trục q
    return torch.stack([res_d, res_q], dim=1)  # Gộp thành tensor (N,2)


class PINNModel(BaselineMLP):  # Lớp mô hình PINN
    """PINN kế thừa kiến trúc MLP và mang Φm cố định như tri thức vật lý."""  # Mô tả lớp

    def __init__(self, in_scale, out_lo, out_hi, phi_m, hidden_sizes=(32, 32), activation="tanh"):  # Hàm khởi tạo
        """Dựng mạng và đăng ký Φm làm hằng số không huấn luyện."""  # Mô tả hàm
        super().__init__(in_scale, out_lo, out_hi, hidden_sizes, activation)  # Dựng phần mạng cơ sở
        self.register_buffer("phi_m", torch.tensor(float(phi_m)))  # Từ thông nam châm cố định

    def residual(self, x, aux):  # Tính sai số vật lý
        """Dự đoán tham số rồi trả về sai số phương trình điện áp (V)."""  # Mô tả hàm
        return pmsm_voltage_residual(self.forward(x), aux, self.phi_m)  # Gọi hàm sai số


def build_pinn(cfg, stats):  # Hàm dựng PINN từ cấu hình
    """Dựng PINNModel từ cấu hình và thống kê nhãn."""  # Mô tả hàm
    m = cfg["motor"]  # Nhóm thông số động cơ
    u_base = m["u_dc"] / 3 ** 0.5  # Điện áp cơ sở (V)
    in_scale = [u_base, u_base, m["i_rated_peak"], m["i_rated_peak"], m["n_max"]]  # Hệ số chuẩn hóa đầu vào
    lo = [stats[k]["lo"] for k in ("Rs", "Ld", "Lq")]  # Cận dưới đầu ra
    hi = [stats[k]["hi"] for k in ("Rs", "Ld", "Lq")]  # Cận trên đầu ra
    return PINNModel(in_scale, lo, hi, m["phi_m"], cfg["model"]["hidden_sizes"], cfg["model"]["activation"])  # Dựng PINN
