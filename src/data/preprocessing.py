"""Tiền xử lý dữ liệu động cơ PMSM: kiểm tra, tính đạo hàm dòng điện, gắn cờ vận hành."""  # Mô tả mô-đun
import math  # Hàm toán học cơ bản
import numpy as np  # Thư viện số học mảng
import pandas as pd  # Thư viện bảng dữ liệu
from scipy.signal import savgol_filter  # Bộ lọc Savitzky-Golay

REQUIRED = ["Time", "Id", "Iq", "ud", "uq", "nm", "Rs", "Ld", "Lq"]  # Cột bắt buộc


def load_raw_csv(path, columns=None):  # Hàm đọc tệp thô
    """Đọc tệp CSV thô, kiểm tra cột bắt buộc và đơn vị điện cảm."""  # Mô tả hàm
    df = pd.read_csv(path)  # Đọc tệp CSV
    df.columns = [c.strip() for c in df.columns]  # Loại khoảng trắng ở tên cột
    missing = [c for c in REQUIRED if c not in df.columns]  # Tìm cột còn thiếu
    if missing:  # Nếu thiếu cột
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")  # Báo lỗi rõ ràng
    if float(df["Ld"].median()) > 0.1:  # Nếu điện cảm lớn bất thường thì đơn vị là mH
        df[["Ld", "Lq"]] = df[["Ld", "Lq"]] * 1e-3  # Đổi mH sang H
    return df  # Trả về bảng dữ liệu


def check_time_grid(df):  # Hàm kiểm tra lưới thời gian
    """Tính chu kỳ lấy mẫu và độ không đều của lưới thời gian."""  # Mô tả hàm
    t = df["Time"].to_numpy(dtype=float)  # Mảng thời gian
    d = np.diff(t)  # Hiệu thời gian giữa các mẫu liên tiếp
    dt = float(np.median(d))  # Chu kỳ lấy mẫu danh định
    jitter = float(np.max(np.abs(d - dt)) / dt)  # Độ lệch tương đối lớn nhất
    return {"dt": dt, "fs": 1.0 / dt, "jitter": jitter, "n": int(len(t))}  # Trả về thống kê


def compute_omega_e(nm, pole_pairs):  # Hàm tính tốc độ góc điện
    """Đổi tốc độ cơ nm (vòng/phút) sang tốc độ góc điện ωe (rad/s)."""  # Mô tả hàm
    return np.asarray(nm, dtype=float) * 2.0 * math.pi * pole_pairs / 60.0  # ωe = p·2π·n/60


def compute_derivatives(df, dt, window, polyorder):  # Hàm tính đạo hàm dòng điện
    """Tính did/dt và diq/dt bằng bộ lọc Savitzky-Golay."""  # Mô tả hàm
    out = {}  # Từ điển kết quả
    for col, name in (("Id", "did_dt"), ("Iq", "diq_dt")):  # Duyệt hai trục
        out[name] = savgol_filter(df[col].to_numpy(dtype=float), window, polyorder, deriv=1, delta=dt, mode="interp")  # Đạo hàm theo thời gian
    return out  # Trả về hai mảng đạo hàm


def voltage_residual_numpy(df, phi_m, pole_pairs):  # Hàm sai số phương trình điện áp
    """Tính sai số phương trình điện áp d-q với tham số nhãn trong bộ dữ liệu."""  # Mô tả hàm
    w = df["omega_e"].to_numpy()  # Tốc độ góc điện
    rd = df["ud"] - (df["Rs"] * df["Id"] + df["Ld"] * df["did_dt"] - w * df["Lq"] * df["Iq"])  # Sai số trục d
    rq = df["uq"] - (df["Rs"] * df["Iq"] + df["Lq"] * df["diq_dt"] + w * df["Ld"] * df["Id"] + w * phi_m)  # Sai số trục q
    return rd.to_numpy(), rq.to_numpy()  # Trả về hai mảng sai số


def flag_rs_transient(df, threshold, t_settle):  # Hàm gắn cờ quá độ Rs
    """Gắn cờ quá độ cho các mẫu nằm trong t_settle sau mỗi bước nhảy của Rs."""  # Mô tả hàm
    rs = df["Rs"].to_numpy(dtype=float)  # Mảng Rs nhãn
    t = df["Time"].to_numpy(dtype=float)  # Mảng thời gian
    rel = np.abs(np.diff(rs)) / np.maximum(np.abs(rs[:-1]), 1e-12)  # Thay đổi tương đối giữa hai mẫu
    idx = np.where(rel > threshold)[0] + 1  # Chỉ số các mẫu ngay sau bước nhảy
    flag = np.zeros(len(rs), dtype=bool)  # Mảng cờ ban đầu bằng sai
    for i in idx:  # Duyệt từng bước nhảy
        flag |= (t >= t[i]) & (t < t[i] + t_settle)  # Gắn cờ cửa sổ sau bước nhảy
    return flag, int(len(idx))  # Trả về cờ và số bước nhảy


def flag_saturation(df, ld_unsat, lq_unsat, ratio):  # Hàm gắn cờ bão hòa từ
    """Gắn cờ bão hòa khi Ld hoặc Lq nhỏ hơn ratio lần giá trị chưa bão hòa."""  # Mô tả hàm
    return ((df["Ld"] < ratio * ld_unsat) | (df["Lq"] < ratio * lq_unsat)).to_numpy()  # Điều kiện bão hòa


def build_processed(df, cfg):  # Hàm dựng bảng đã xử lý
    """Dựng bảng đã xử lý gồm tín hiệu, đạo hàm, ωe và các cờ vận hành."""  # Mô tả hàm
    d, m = cfg["data"], cfg["motor"]  # Rút gọn các nhóm cấu hình
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)  # Loại giá trị không hợp lệ
    grid = check_time_grid(df)  # Kiểm tra lưới thời gian
    der = compute_derivatives(df, grid["dt"], d["sg_window"], d["sg_polyorder"])  # Tính đạo hàm
    out = df.copy()  # Sao chép bảng gốc
    out["did_dt"], out["diq_dt"] = der["did_dt"], der["diq_dt"]  # Thêm cột đạo hàm
    out["omega_e"] = compute_omega_e(out["nm"], m["pole_pairs"])  # Thêm cột ωe
    out["is_transient"], _ = flag_rs_transient(out, d["rs_step_threshold"], d["t_settle"])  # Thêm cờ quá độ
    out["is_saturated"] = flag_saturation(out, m["ld_unsat"], m["lq_unsat"], d["sat_ratio"])  # Thêm cờ bão hòa
    half = d["sg_window"] // 2  # Số mẫu biên bị ảnh hưởng bởi bộ lọc
    out = out.iloc[half:len(out) - half].reset_index(drop=True)  # Cắt các mẫu biên
    return out.iloc[::d["stride"]].reset_index(drop=True)  # Lấy mẫu con theo bước stride


def label_stats(df_list, margin):  # Hàm thống kê nhãn
    """Tính khoảng giá trị nhãn trên tập huấn luyện, mở rộng theo biên margin."""  # Mô tả hàm
    allv = pd.concat(df_list, ignore_index=True)  # Gộp các bảng huấn luyện
    stats = {}  # Từ điển kết quả
    for c in ("Rs", "Ld", "Lq"):  # Duyệt ba tham số
        lo, hi = float(allv[c].min()), float(allv[c].max())  # Cực tiểu và cực đại
        stats[c] = {"min": lo, "max": hi, "lo": lo * (1 - margin), "hi": hi * (1 + margin)}  # Khoảng mở rộng
    return stats  # Trả về thống kê
