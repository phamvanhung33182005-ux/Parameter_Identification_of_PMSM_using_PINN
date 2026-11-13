"""Lớp Dataset và hàm dựng DataLoader cho MLP nhận dạng Ld, Lq theo (id, iq)."""  # Mô tả mô-đun
import os  # Thao tác đường dẫn
import numpy as np  # Thư viện số học mảng
import pandas as pd  # Thư viện bảng dữ liệu
import torch  # Thư viện học sâu
from torch.utils.data import Dataset, DataLoader  # Lớp cơ sở và bộ nạp dữ liệu

INPUT_COLS = ["Id", "Iq"]  # Đầu vào của mạng (A)
LABEL_COLS = ["Ld", "Lq"]  # Nhãn của mạng (H)


class PMSMDataset(Dataset):  # Lớp tập dữ liệu
    """Tập dữ liệu PMSM: đầu vào (id, iq) và nhãn (Ld, Lq) theo đơn vị SI."""  # Mô tả lớp

    def __init__(self, frames):  # Hàm khởi tạo
        """Nhận danh sách bảng đã xử lý và dựng các tensor."""  # Mô tả hàm
        df = pd.concat(frames, ignore_index=True)  # Gộp các bảng
        self.x = torch.tensor(df[INPUT_COLS].to_numpy(dtype=np.float32))  # Tensor đầu vào
        self.y = torch.tensor(df[LABEL_COLS].to_numpy(dtype=np.float32))  # Tensor nhãn
        self.flags = df[["is_transient"]].to_numpy(dtype=bool)  # Cờ vận hành (quá độ), chỉ để thống kê

    def __len__(self):  # Số mẫu
        """Trả về số mẫu của tập dữ liệu."""  # Mô tả hàm
        return self.x.shape[0]  # Số hàng của tensor đầu vào

    def __getitem__(self, i):  # Lấy một mẫu
        """Trả về từ điển gồm x, y của mẫu thứ i."""  # Mô tả hàm
        return {"x": self.x[i], "y": self.y[i]}  # Gói mẫu


def load_processed(processed_dir, names):  # Hàm nạp các tệp đã xử lý
    """Nạp danh sách tệp đã xử lý theo tên tệp thô."""  # Mô tả hàm
    frames = []  # Danh sách bảng
    for n in names:  # Duyệt từng tên tệp
        stem = os.path.splitext(n)[0]  # Bỏ phần mở rộng
        frames.append(pd.read_csv(os.path.join(processed_dir, f"{stem}_processed.csv.gz")))  # Đọc tệp đã xử lý
    return frames  # Trả về danh sách bảng


def build_dataloaders(cfg, processed_dir, num_workers=0):  # Hàm dựng bộ nạp dữ liệu
    """Dựng DataLoader cho tập huấn luyện, kiểm định và kiểm thử theo cấu hình."""  # Mô tả hàm
    sp, bs = cfg["data"]["split"], cfg["train"]["batch_size"]  # Cách chia tập và kích thước lô
    sets = {k: PMSMDataset(load_processed(processed_dir, sp[k])) for k in ("train", "val", "test")}  # Dựng ba tập
    g = torch.Generator().manual_seed(cfg["project"]["seed"])  # Bộ sinh ngẫu nhiên cố định
    loaders = {  # Từ điển bộ nạp
        "train": DataLoader(sets["train"], batch_size=bs, shuffle=True, generator=g, num_workers=num_workers, drop_last=False),  # Trộn mẫu khi huấn luyện
        "val": DataLoader(sets["val"], batch_size=bs, shuffle=False, num_workers=num_workers),  # Giữ thứ tự khi kiểm định
        "test": DataLoader(sets["test"], batch_size=bs, shuffle=False, num_workers=num_workers),  # Giữ thứ tự khi kiểm thử
    }  # Kết thúc từ điển
    return sets, loaders  # Trả về tập và bộ nạp
