"""Kịch bản kiểm thử: nạp mô hình đã huấn luyện, đánh giá trên tập kiểm thử và xuất kết quả."""  # Mô tả mô-đun
import argparse  # Đọc tham số dòng lệnh
import json  # Ghi kết quả JSON
import os  # Thao tác đường dẫn
import numpy as np  # Thư viện số học mảng
import pandas as pd  # Thư viện bảng dữ liệu
import torch  # Thư viện học sâu
import matplotlib  # Thư viện vẽ đồ thị
matplotlib.use("Agg")  # Dùng nền không giao diện
import matplotlib.pyplot as plt  # Giao diện vẽ đồ thị
from src.data.dataset import load_processed, INPUT_COLS, LABEL_COLS, AUX_COLS  # Hàm và hằng số dữ liệu
from src.evaluation import metrics as mt  # Các chỉ số đánh giá
from src.models.baselineMLP import BaselineMLP  # Mạng cơ sở
from src.models.pinn_model import PINNModel  # Mạng PINN
from src.utils.common import load_config, get_paths  # Tiện ích cấu hình


def load_model(ckpt_path, kind, phi_m):  # Nạp mô hình từ điểm lưu
    """Dựng lại mô hình từ kiến trúc đã lưu và nạp trọng số."""  # Mô tả hàm
    ck = torch.load(ckpt_path, map_location="cpu")  # Đọc điểm lưu
    a = ck["arch"]  # Lấy thông tin kiến trúc
    if kind == "pinn":  # Trường hợp PINN
        m = PINNModel(a["in_scale"], a["out_lo"], a["out_hi"], phi_m, a["hidden_sizes"], a["activation"])  # Dựng PINN
    else:  # Trường hợp mạng cơ sở
        m = BaselineMLP(a["in_scale"], a["out_lo"], a["out_hi"], a["hidden_sizes"], a["activation"])  # Dựng MLP
    m.load_state_dict(ck["state_dict"])  # Nạp trọng số
    return m.eval()  # Chuyển sang chế độ suy luận


def run_test(cfg, kind, split="test"):  # Chạy kiểm thử
    """Đánh giá mô hình kind trên tập split, lưu JSON và đồ thị, trả về kết quả."""  # Mô tả hàm
    P, ev, mo = get_paths(cfg), cfg["evaluation"], cfg["motor"]  # Đường dẫn và nhóm cấu hình
    model = load_model(os.path.join(P["models"], f"{kind}_best.pt"), kind, mo["phi_m"])  # Nạp mô hình
    frames = load_processed(P["processed"], cfg["data"]["split"][split])  # Nạp các tệp của tập cần đánh giá
    preds, trues, flags, auxs, rec = [], [], [], [], {"d": [], "q": [], "md": [], "mq": []}  # Bộ tích lũy
    for df in frames:  # Duyệt từng tệp
        x = torch.tensor(df[INPUT_COLS].to_numpy(np.float32))  # Đầu vào
        with torch.no_grad():  # Tắt gradient
            p = model(x).numpy().astype(np.float64)  # Dự đoán tham số
        a = df[AUX_COLS].to_numpy(np.float64)  # Đại lượng phụ
        dt = float(np.median(np.diff(df["Time"].to_numpy())))  # Chu kỳ giữa hai mẫu
        n_h = max(2, int(round(ev["recon_horizon_s"] / dt)))  # Số bước mỗi đoạn mô phỏng
        rd, rq, md, mq = mt.reconstruct_currents(p, a, dt, n_h, mo["phi_m"])  # Tái tạo dòng điện
        for k, v in zip(("d", "q", "md", "mq"), (rd, rq, md, mq)):  # Duyệt bốn mảng
            rec[k].append(v.ravel())  # Làm phẳng và lưu
        preds.append(p), trues.append(df[LABEL_COLS].to_numpy(np.float64))  # Lưu dự đoán và nhãn
        flags.append(df[["is_transient", "is_saturated"]].to_numpy(bool)), auxs.append(a)  # Lưu cờ và đại lượng phụ
    pred, true, flg, aux = np.vstack(preds), np.vstack(trues), np.vstack(flags), np.vstack(auxs)  # Gộp các tệp
    masks = mt.condition_masks(flg, aux[:, 2], aux[:, 3], mo["i_rated_peak"], ev["i_range_low"])  # Mặt nạ điều kiện
    rows = mt.evaluate_parameters(pred, true, masks, ev["tolerances"], ev["verdict_stat"])  # Bảng sai số
    cat = {k: np.concatenate(v) for k, v in rec.items()}  # Gộp dòng điện các tệp
    rmse = {"id_pct": mt.current_rmse_pct(cat["d"], cat["md"], mo["i_rated_peak"]), "iq_pct": mt.current_rmse_pct(cat["q"], cat["mq"], mo["i_rated_peak"])}  # RMSE dòng điện
    rmse["pass"] = bool(max(rmse["id_pct"], rmse["iq_pct"]) <= ev["rmse_limit_pct"])  # Kết luận RMSE
    out = {"model": kind, "split": split, "parameters": rows, "current_rmse": rmse}  # Gói kết quả
    os.makedirs(P["results"], exist_ok=True)  # Tạo thư mục kết quả
    with open(os.path.join(P["results"], f"test_{kind}_{split}.json"), "w", encoding="utf-8") as f:  # Mở tệp JSON
        json.dump(out, f, ensure_ascii=False, indent=1)  # Ghi kết quả
    plot_tracking(pred, true, os.path.join(P["results"], f"tracking_{kind}_{split}.png"))  # Vẽ đồ thị bám nhãn
    return out  # Trả về kết quả


def plot_tracking(pred, true, path, max_points=4000):  # Vẽ đồ thị so sánh
    """Vẽ giá trị dự đoán so với nhãn của Rs, Ld, Lq theo chỉ số mẫu."""  # Mô tả hàm
    idx = np.linspace(0, len(pred) - 1, min(max_points, len(pred))).astype(int)  # Chỉ số lấy mẫu con
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)  # Ba đồ thị chồng dọc
    for j, (name, unit) in enumerate((("Rs", "Ω"), ("Ld", "H"), ("Lq", "H"))):  # Duyệt ba tham số
        ax[j].plot(idx, true[idx, j], label="Nhãn")  # Đường nhãn
        ax[j].plot(idx, pred[idx, j], label="Dự đoán", linewidth=0.8)  # Đường dự đoán
        ax[j].set_ylabel(f"{name} ({unit})")  # Nhãn trục tung
        ax[j].legend()  # Chú giải
    ax[-1].set_xlabel("Chỉ số mẫu")  # Nhãn trục hoành
    fig.tight_layout()  # Căn chỉnh bố cục
    fig.savefig(path, dpi=150)  # Lưu ảnh
    plt.close(fig)  # Giải phóng bộ nhớ


def main():  # Hàm dòng lệnh
    """Chạy kiểm thử từ dòng lệnh: python -m src.evaluation.test --kind pinn."""  # Mô tả hàm
    ap = argparse.ArgumentParser()  # Bộ phân tích tham số
    ap.add_argument("--config", default="configs/config.yaml")  # Đường dẫn cấu hình
    ap.add_argument("--kind", default="pinn", choices=["pinn", "baseline"])  # Loại mô hình
    ap.add_argument("--split", default="test")  # Tập đánh giá
    a = ap.parse_args()  # Đọc tham số
    res = run_test(load_config(a.config), a.kind, a.split)  # Chạy kiểm thử
    print(json.dumps(res["current_rmse"], ensure_ascii=False))  # In RMSE dòng điện


if __name__ == "__main__":  # Điểm vào khi chạy trực tiếp
    main()  # Gọi hàm chính
