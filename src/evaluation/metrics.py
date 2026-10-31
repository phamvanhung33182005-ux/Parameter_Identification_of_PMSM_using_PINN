"""Chỉ số đánh giá: sai số tương đối từng tham số theo điều kiện và RMSE dòng điện tái tạo."""  # Mô tả mô-đun
import numpy as np  # Thư viện số học mảng


def relative_error_pct(pred, true):  # Sai số tương đối
    """Trả về sai số tương đối tuyệt đối theo phần trăm."""  # Mô tả hàm
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-12) * 100.0  # |ŷ−y|/|y|·100


def error_stats(err):  # Thống kê sai số
    """Trả về trung bình, RMS, P95, P99 và cực đại của mảng sai số."""  # Mô tả hàm
    if len(err) == 0:  # Không có mẫu
        return {"n": 0, "mean": np.nan, "rms": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}  # Trả về rỗng
    return {"n": int(len(err)), "mean": float(err.mean()), "rms": float(np.sqrt((err ** 2).mean())), "p95": float(np.percentile(err, 95)), "p99": float(np.percentile(err, 99)), "max": float(err.max())}  # Các thống kê


def condition_masks(flags, i_d, i_q, i_rated, low):  # Mặt nạ điều kiện đánh giá
    """Dựng mặt nạ theo điều kiện xác định trong PRD."""  # Mô tả hàm
    trans, sat = flags[:, 0], flags[:, 1]  # Cờ quá độ và bão hòa
    return {  # Từ điển mặt nạ
        "Rs_steady": ~trans, "Rs_transient": trans,  # Điều kiện của Rs
        "Ld_unsat": (~sat) & (np.abs(i_d) >= low * i_rated), "Ld_sat": sat & (np.abs(i_d) >= low * i_rated),  # Điều kiện của Ld
        "Lq_unsat": (~sat) & (np.abs(i_q) >= low * i_rated), "Lq_sat": sat & (np.abs(i_q) >= low * i_rated),  # Điều kiện của Lq
    }  # Kết thúc từ điển


def evaluate_parameters(pred, true, masks, tolerances, verdict_stat="max"):  # Đánh giá tham số
    """Tính thống kê sai số và kết luận đạt/không đạt cho từng tham số và điều kiện."""  # Mô tả hàm
    rows = []  # Danh sách kết quả
    for j, p in enumerate(("Rs", "Ld", "Lq")):  # Duyệt ba tham số
        err = relative_error_pct(pred[:, j], true[:, j])  # Sai số tương đối từng mẫu
        for cond, lim in tolerances[p].items():  # Duyệt từng điều kiện
            s = error_stats(err[masks[f"{p}_{cond}"]])  # Thống kê trong điều kiện
            ok = bool(s["n"] > 0 and s[verdict_stat] <= lim)  # Kết luận theo thống kê đã chọn
            rows.append({"param": p, "condition": cond, "limit_pct": lim, **s, "pass": ok})  # Ghi một hàng
    return rows  # Trả về bảng kết quả


def reconstruct_currents(pred, aux, dt, horizon_steps, phi_m):  # Tái tạo dòng điện
    """Mô phỏng id, iq theo mô hình điện với tham số nhận dạng và Φm cố định trên các đoạn độc lập."""  # Mô tả hàm
    rs, ld, lq = pred[:, 0], pred[:, 1], pred[:, 2]  # Tách tham số
    ud, uq, i_d, i_q, w = [aux[:, k] for k in range(5)]  # Tách đại lượng đo
    n_seg = len(rs) // horizon_steps  # Số đoạn đầy đủ
    cut = n_seg * horizon_steps  # Số mẫu dùng
    rs, ld, lq, ud, uq, i_d, i_q, w = [a[:cut].reshape(n_seg, horizon_steps) for a in (rs, ld, lq, ud, uq, i_d, i_q, w)]  # Chia thành các đoạn
    sd, sq = i_d[:, 0].copy(), i_q[:, 0].copy()  # Điều kiện đầu lấy từ giá trị đo
    out_d, out_q = np.zeros_like(i_d), np.zeros_like(i_q)  # Mảng kết quả
    for k in range(horizon_steps):  # Duyệt từng bước thời gian
        out_d[:, k], out_q[:, k] = sd, sq  # Ghi trạng thái hiện tại
        dd = (ud[:, k] - rs[:, k] * sd + w[:, k] * lq[:, k] * sq) / ld[:, k]  # Đạo hàm id theo mô hình
        dq = (uq[:, k] - rs[:, k] * sq - w[:, k] * ld[:, k] * sd - w[:, k] * phi_m) / lq[:, k]  # Đạo hàm iq theo mô hình
        sd, sq = sd + dt * dd, sq + dt * dq  # Tích phân Euler
    return out_d, out_q, i_d, i_q  # Trả về dòng tái tạo và dòng đo



def current_rmse_pct(rec, meas, i_base):  # RMSE dòng điện
    """Trả về RMSE chuẩn hóa theo dòng cơ sở (%)."""  # Mô tả hàm
    return float(np.sqrt(np.mean((rec - meas) ** 2)) / i_base * 100.0)  # RMSE chia dòng cơ sở
