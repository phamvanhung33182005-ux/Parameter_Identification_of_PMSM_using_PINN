"""Hàm mất mát: thành phần dữ liệu, thành phần phạt riêng Rs, thành phần vật lý và tổng hợp có tăng dần trọng số."""  # Mô tả mô-đun
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


class RsLoss(nn.Module):  # Thành phần phạt riêng Rs
    """Bình phương sai số Rs dự đoán so với nhãn, chia cho Rs tham chiếu: ((R̂s - Rs)/Rs_ref)²."""  # Mô tả lớp

    def __init__(self, rs_ref):  # Hàm khởi tạo
        """Nhận Rs tham chiếu (Ω) để chuẩn hóa sai số."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.rs_ref = float(rs_ref)  # Lưu Rs tham chiếu

    def forward(self, pred, target):  # Tính mất mát
        """Trả về trung bình bình phương sai số tương đối của Rs (cột 0)."""  # Mô tả hàm
        return (((pred[:, 0] - target[:, 0]) / self.rs_ref) ** 2).mean()  # Chỉ lấy Rs, bình phương, lấy trung bình


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
    """Tổng hợp L = λ_data·L_data + λ_phys(e)·L_phys + λ_rs·L_rs; λ_phys tăng tuyến tính theo kỷ nguyên, λ_rs không đổi."""  # Mô tả lớp

    def __init__(self, ref, phi_m, u_base, lambda_data, lambda_phys, warmup_epochs, lambda_rs=0.0):  # Hàm khởi tạo
        """Nhận các trọng số, số kỷ nguyên tăng dần và trọng số phạt Rs."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.data, self.phys, self.rs = DataLoss(ref), PhysicsLoss(phi_m, u_base), RsLoss(ref[0])  # Ba thành phần (ref[0] là Rs tham chiếu)
        self.lam_d, self.lam_p, self.warm, self.lam_r = lambda_data, lambda_phys, warmup_epochs, lambda_rs  # Lưu trọng số

    def phys_weight(self, epoch):  # Trọng số vật lý theo kỷ nguyên
        """Trả về λ_phys tại kỷ nguyên epoch (bắt đầu từ 0)."""  # Mô tả hàm
        if self.warm <= 0:  # Không dùng giai đoạn tăng dần
            return self.lam_p  # Dùng trọng số đầy đủ
        return self.lam_p * min(1.0, epoch / self.warm)  # Tăng tuyến tính tới giá trị đầy đủ

    def forward(self, pred, batch, epoch=0):  # Tính mất mát tổng
        """Trả về mất mát tổng và từ điển các thành phần."""  # Mô tả hàm
        ld = self.data(pred, batch["y"])  # Thành phần dữ liệu
        lp = self.phys(pred, batch["aux"])  # Thành phần vật lý
        lr = self.rs(pred, batch["y"])  # Thành phần phạt riêng Rs
        total = self.lam_d * ld + self.phys_weight(epoch) * lp + self.lam_r * lr  # Tổng có trọng số
        return total, {"data": float(ld.detach()), "phys": float(lp.detach()), "rs": float(lr.detach())}  # Trả về tổng và chi tiết


def build_loss(cfg, mode="pinn"):  # Hàm dựng mất mát từ cấu hình
    """Dựng mất mát; mode='baseline' bỏ thành phần vật lý (λ_phys = 0), vẫn giữ λ_rs để hai mô hình so sánh công bằng."""  # Mô tả hàm
    m, l = cfg["motor"], cfg["loss"]  # Rút gọn các nhóm cấu hình
    ref = [m["rs_20c"], m["ld_unsat"], m["lq_unsat"]]  # Giá trị tham chiếu chuẩn hóa
    lam_p = 0.0 if mode == "baseline" else l["lambda_phys"]  # Trọng số vật lý theo chế độ
    lam_r = l.get("lambda_rs", 0.0)  # Trọng số phạt Rs, thiếu khóa thì tắt để tương thích cấu hình cũ
    return PINNLoss(ref, m["phi_m"], m["u_dc"] / 3 ** 0.5, l["lambda_data"], lam_p, l["warmup_epochs"], lam_r)  # Dựng mất mát
