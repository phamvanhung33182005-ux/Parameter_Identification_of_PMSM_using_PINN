"""Hàm mất mát: thành phần dữ liệu, thành phần vật lý và tổng hợp có tăng dần trọng số."""  # Mô tả mô-đun
import torch  # Thư viện học sâu
import torch.nn as nn  # Các lớp mạng nơ-ron
from src.models.pinn_model import pmsm_voltage_residual  # Hàm sai số vật lý


class DataLoss(nn.Module):  # Thành phần dữ liệu
    """Sai số bình phương trung bình của tham số sau khi chia cho giá trị tham chiếu."""  # Mô tả lớp

    def __init__(self, ref):  # Hàm khởi tạo
        """Nhận giá trị tham chiếu (Rs, Ld, Lq) để chuẩn hóa sai số."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.register_buffer("ref", torch.tensor(ref, dtype=torch.float32))  # Lưu giá trị tham chiếu

    def forward(self, pred, target):  # Tính mất mát
        """Trả về MSE của sai số tương đối so với tham chiếu."""  # Mô tả hàm
        return (((pred - target) / self.ref) ** 2).mean()  # Bình phương, lấy trung bình


class PhysicsLoss(nn.Module):  # Thành phần vật lý
    """Bình phương trung bình sai số phương trình điện áp, chuẩn hóa theo điện áp cơ sở."""  # Mô tả lớp

    def __init__(self, phi_m, u_base):  # Hàm khởi tạo
        """Nhận Φm cố định và điện áp cơ sở."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.phi_m, self.u_base = float(phi_m), float(u_base)  # Lưu hằng số

    def forward(self, pred, aux):  # Tính mất mát
        """Trả về trung bình bình phương sai số điện áp chuẩn hóa."""  # Mô tả hàm
        res = pmsm_voltage_residual(pred, aux, self.phi_m) / self.u_base  # Sai số chuẩn hóa
        return (res ** 2).mean()  # Bình phương, lấy trung bình


class PINNLoss(nn.Module):  # Mất mát tổng hợp
    """Tổng hợp L = λ_data·L_data + λ_phys(e)·L_phys với λ_phys tăng tuyến tính theo kỷ nguyên."""  # Mô tả lớp

    def __init__(self, ref, phi_m, u_base, lambda_data, lambda_phys, warmup_epochs):  # Hàm khởi tạo
        """Nhận các trọng số và số kỷ nguyên tăng dần."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.data, self.phys = DataLoss(ref), PhysicsLoss(phi_m, u_base)  # Hai thành phần
        self.lam_d, self.lam_p, self.warm = lambda_data, lambda_phys, warmup_epochs  # Lưu trọng số

    def phys_weight(self, epoch):  # Trọng số vật lý theo kỷ nguyên
        """Trả về λ_phys tại kỷ nguyên epoch (bắt đầu từ 0)."""  # Mô tả hàm
        if self.warm <= 0:  # Không dùng giai đoạn tăng dần
            return self.lam_p  # Dùng trọng số đầy đủ
        return self.lam_p * min(1.0, epoch / self.warm)  # Tăng tuyến tính tới giá trị đầy đủ

    def forward(self, pred, batch, epoch=0):  # Tính mất mát tổng
        """Trả về mất mát tổng và từ điển các thành phần."""  # Mô tả hàm
        ld = self.data(pred, batch["y"])  # Thành phần dữ liệu
        lp = self.phys(pred, batch["aux"])  # Thành phần vật lý
        total = self.lam_d * ld + self.phys_weight(epoch) * lp  # Tổng có trọng số
        return total, {"data": float(ld.detach()), "phys": float(lp.detach())}  # Trả về tổng và chi tiết


def build_loss(cfg, mode="pinn"):  # Hàm dựng mất mát từ cấu hình
    """Dựng mất mát; mode='baseline' bỏ thành phần vật lý (λ_phys = 0)."""  # Mô tả hàm
    m, l = cfg["motor"], cfg["loss"]  # Rút gọn các nhóm cấu hình
    ref = [m["rs_20c"], m["ld_unsat"], m["lq_unsat"]]  # Giá trị tham chiếu chuẩn hóa
    lam_p = 0.0 if mode == "baseline" else l["lambda_phys"]  # Trọng số vật lý theo chế độ
    return PINNLoss(ref, m["phi_m"], m["u_dc"] / 3 ** 0.5, l["lambda_data"], lam_p, l["warmup_epochs"])  # Dựng mất mát
