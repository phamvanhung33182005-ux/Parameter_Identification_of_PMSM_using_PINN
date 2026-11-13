"""Hàm mất mát và bộ huấn luyện MLP nhận dạng Ld, Lq: Adam, cắt gradient, dừng sớm, lưu điểm tốt nhất."""  # Mô tả mô-đun
import json  # Ghi nhật ký JSON
import os  # Thao tác đường dẫn
import time  # Đo thời gian huấn luyện
import torch  # Thư viện học sâu
import torch.nn as nn  # Các lớp mạng nơ-ron


class RelativeMSE(nn.Module):  # Hàm mất mát
    """Sai số bình phương trung bình của (Ld, Lq) sau khi chia cho giá trị tham chiếu."""  # Mô tả lớp

    def __init__(self, ref):  # Hàm khởi tạo
        """Nhận giá trị tham chiếu (Ld, Lq) để chuẩn hóa sai số."""  # Mô tả hàm
        super().__init__()  # Gọi lớp cha
        self.register_buffer("ref", torch.tensor(ref, dtype=torch.float32))  # Lưu giá trị tham chiếu

    def forward(self, pred, target):  # Tính mất mát
        """Trả về MSE của sai số tương đối so với tham chiếu."""  # Mô tả hàm
        return (((pred - target) / self.ref) ** 2).mean()  # Bình phương, lấy trung bình


def build_loss(cfg):  # Hàm dựng mất mát từ cấu hình
    """Dựng RelativeMSE với tham chiếu (ld_ref, lq_ref) của cấu hình."""  # Mô tả hàm
    m = cfg["motor"]  # Nhóm thông số động cơ
    return RelativeMSE([m["ld_ref"], m["lq_ref"]])  # Dựng mất mát


class Trainer:  # Lớp bộ huấn luyện
    """Huấn luyện mô hình với Adam, cắt gradient, giảm tốc độ học và dừng sớm."""  # Mô tả lớp

    def __init__(self, model, loss_fn, loaders, cfg, device, save_dir, name):  # Hàm khởi tạo
        """Nhận mô hình, mất mát, bộ nạp dữ liệu, cấu hình và nơi lưu."""  # Mô tả hàm
        t = cfg["train"]  # Nhóm cấu hình huấn luyện
        self.model, self.loss_fn, self.loaders = model.to(device), loss_fn.to(device), loaders  # Lưu đối tượng chính
        self.cfg, self.device, self.name = cfg, device, name  # Lưu cấu hình, thiết bị, tên
        self.opt = torch.optim.Adam(self.model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])  # Bộ tối ưu Adam
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=max(1, t["patience"] // 3))  # Giảm tốc độ học khi chững
        self.ckpt = os.path.join(save_dir, f"{name}_best.pt")  # Đường dẫn điểm tốt nhất
        self.history = []  # Nhật ký các kỷ nguyên

    def _run_epoch(self, split):  # Chạy một kỷ nguyên
        """Chạy một kỷ nguyên trên tập split và trả về mất mát trung bình."""  # Mô tả hàm
        train = split == "train"  # Cờ chế độ huấn luyện
        self.model.train(train)  # Bật hoặc tắt chế độ huấn luyện
        tot, n = 0.0, 0  # Bộ tích lũy
        for batch in self.loaders[split]:  # Duyệt từng lô
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)  # Chuyển lô sang thiết bị
            with torch.set_grad_enabled(train):  # Chỉ tính gradient khi huấn luyện
                loss = self.loss_fn(self.model(x), y)  # Lan truyền thuận và tính mất mát
            if train:  # Nếu đang huấn luyện
                self.opt.zero_grad()  # Xóa gradient cũ
                loss.backward()  # Lan truyền ngược
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"]["grad_clip"])  # Cắt gradient
                self.opt.step()  # Cập nhật trọng số
            k = x.shape[0]  # Kích thước lô
            tot, n = tot + float(loss.detach()) * k, n + k  # Cộng dồn mất mát
        return tot / n  # Trả về trung bình

    def fit(self):  # Vòng huấn luyện chính
        """Huấn luyện tới khi hết kỷ nguyên hoặc dừng sớm; trả về nhật ký."""  # Mô tả hàm
        t, best, wait, t0 = self.cfg["train"], float("inf"), 0, time.time()  # Trạng thái ban đầu
        for ep in range(t["epochs"]):  # Duyệt từng kỷ nguyên
            tr = self._run_epoch("train")  # Huấn luyện
            va = self._run_epoch("val")  # Kiểm định
            self.sched.step(va)  # Cập nhật tốc độ học
            self.history.append({"epoch": ep, "train": tr, "val": va, "lr": self.opt.param_groups[0]["lr"]})  # Ghi nhật ký
            if va < best:  # Nếu cải thiện
                best, wait = va, 0  # Đặt lại bộ đếm
                self.save()  # Lưu điểm tốt nhất
            else:  # Nếu không cải thiện
                wait += 1  # Tăng bộ đếm chờ
            if wait >= t["patience"]:  # Hết kiên nhẫn
                break  # Dừng sớm
        if not os.path.exists(self.ckpt):  # Nếu chưa có điểm lưu nào
            self.save()  # Lưu trạng thái cuối
        return {"history": self.history, "best_val": best, "seconds": time.time() - t0}  # Trả về kết quả

    def save(self):  # Lưu điểm kiểm tra
        """Lưu trọng số và cấu hình kiến trúc."""  # Mô tả hàm
        torch.save({"state_dict": self.model.state_dict(), "arch": self.model.get_config()}, self.ckpt)  # Ghi tệp .pt

    def save_history(self, path):  # Lưu nhật ký
        """Ghi nhật ký huấn luyện ra tệp JSON."""  # Mô tả hàm
        with open(path, "w", encoding="utf-8") as f:  # Mở tệp ghi
            json.dump(self.history, f, ensure_ascii=False, indent=1)  # Ghi JSON
