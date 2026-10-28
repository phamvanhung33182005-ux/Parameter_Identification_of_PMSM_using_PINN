"""Mô-đun tiện ích dùng chung: nạp cấu hình, đặt hạt giống, dựng đường dẫn."""  # Mô tả mô-đun
import os  # Thao tác đường dẫn hệ điều hành
import random  # Bộ sinh số ngẫu nhiên chuẩn của Python
import numpy as np  # Thư viện số học mảng
import yaml  # Đọc tệp cấu hình YAML


def load_config(path):  # Hàm nạp tệp cấu hình
    """Đọc tệp YAML và trả về từ điển cấu hình."""  # Mô tả hàm
    with open(path, "r", encoding="utf-8") as f:  # Mở tệp ở chế độ đọc UTF-8
        return yaml.safe_load(f)  # Phân tích YAML an toàn rồi trả về


def set_seed(seed):  # Hàm đặt hạt giống ngẫu nhiên
    """Đặt hạt giống cho random, numpy và torch để tái lập kết quả."""  # Mô tả hàm
    random.seed(seed)  # Hạt giống cho random
    np.random.seed(seed)  # Hạt giống cho numpy
    os.environ["PYTHONHASHSEED"] = str(seed)  # Cố định băm chuỗi
    try:  # Thử nạp torch nếu có
        import torch  # Thư viện học sâu
        torch.manual_seed(seed)  # Hạt giống cho torch CPU
        torch.cuda.manual_seed_all(seed)  # Hạt giống cho mọi GPU
        torch.backends.cudnn.deterministic = True  # Buộc thuật toán xác định
        torch.backends.cudnn.benchmark = False  # Tắt tự chọn thuật toán
    except ImportError:  # Trường hợp chưa cài torch
        pass  # Bỏ qua


def get_paths(cfg):  # Hàm dựng đường dẫn tuyệt đối
    """Trả về từ điển đường dẫn tuyệt đối trên Drive từ cấu hình."""  # Mô tả hàm
    root = cfg["paths"]["drive_root"]  # Thư mục gốc dự án trên Drive
    keys = ["raw", "processed", "models", "results", "backups"]  # Các thư mục con
    return {k: os.path.join(root, cfg["paths"][k]) for k in keys}  # Ghép đường dẫn tuyệt đối
