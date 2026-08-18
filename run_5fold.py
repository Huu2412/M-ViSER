"""
run_5fold.py — Chạy 5-Fold Cross Validation chỉ dùng train.py (không cần evaluate.py)

Cách dùng:
    python run_5fold.py                          # chạy cả 5 folds
    python run_5fold.py --folds 1 2 3            # chỉ chạy fold 1, 2, 3
    python run_5fold.py --config config/config.yaml
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from datetime import datetime

def run_fold(fold: int, config_path: str, base_output_dir: str):
    """Chạy train.py cho 1 fold và trả về kết quả tốt nhất."""
    fold_output_dir = os.path.join(base_output_dir, f"fold_{fold}")
    os.makedirs(fold_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FOLD {fold}/5  —  Output: {fold_output_dir}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train.py",
        "--config", config_path,
        "--override",
        f"dataset.current_fold={fold}",
        f"paths.output_dir={fold_output_dir}",
        f"paths.log_dir={os.path.join(base_output_dir, f'logs_fold{fold}')}",
    ]

    print(f"Lệnh: {' '.join(cmd)}\n")

    # Chạy training — stream output ra terminal
    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if proc.returncode != 0:
        print(f"\n[FOLD {fold}] Training thất bại (exit code {proc.returncode})")
        return None

    # Đọc kết quả từ config.json được save bởi train.py
    best_ckpt = os.path.join(fold_output_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        print(f"\n[FOLD {fold}] Không tìm thấy best_model.pt")
        return None

    # Đọc val metrics từ checkpoint
    try:
        import torch
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        val_metrics = ckpt.get("val_metrics", {})
        epoch = ckpt.get("epoch", "?")
        print(f"\n[FOLD {fold}] ✅ Best checkpoint tại epoch {epoch}:")
        for k, v in val_metrics.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
        return val_metrics
    except Exception as e:
        print(f"[FOLD {fold}] Không đọc được metrics: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="5-Fold CV chỉ dùng train.py")
    parser.add_argument("--config", default="config/config.yaml", help="Đường dẫn config YAML")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Danh sách fold cần chạy (mặc định: 1 2 3 4 5)")
    parser.add_argument("--output_dir", default="checkpoints_5fold",
                        help="Thư mục chứa kết quả tất cả các fold")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n🚀 Bắt đầu 5-Fold Cross Validation")
    print(f"   Config   : {args.config}")
    print(f"   Folds    : {args.folds}")
    print(f"   Output   : {args.output_dir}")
    print(f"   Bắt đầu : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    for fold in args.folds:
        fold_metrics = run_fold(fold, args.config, args.output_dir)
        if fold_metrics is not None:
            all_results[f"fold_{fold}"] = fold_metrics

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{'='*60}")
    print(f"  TỔNG KẾT — {len(all_results)}/{len(args.folds)} folds thành công")
    print(f"{'='*60}")

    if all_results:
        # Tính trung bình các metrics
        metric_keys = list(next(iter(all_results.values())).keys())
        print(f"\n{'Fold':<10}", end="")
        for k in metric_keys:
            print(f"{k:<18}", end="")
        print()
        print("-" * (10 + 18 * len(metric_keys)))

        averages = {k: [] for k in metric_keys}
        for fold_name, metrics in all_results.items():
            print(f"{fold_name:<10}", end="")
            for k in metric_keys:
                v = metrics.get(k, 0)
                averages[k].append(v)
                print(f"{v:<18.4f}" if isinstance(v, float) else f"{v:<18}", end="")
            print()

        print("-" * (10 + 18 * len(metric_keys)))
        print(f"{'Average':<10}", end="")
        for k in metric_keys:
            vals = [v for v in averages[k] if isinstance(v, float)]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"{avg:<18.4f}", end="")
        print()

        # Lưu kết quả tổng hợp
        summary = {
            "folds": all_results,
            "averages": {
                k: sum(v for v in averages[k] if isinstance(v, float)) / max(len([v for v in averages[k] if isinstance(v, float)]), 1)
                for k in metric_keys
            },
            "elapsed_minutes": round(elapsed, 1),
            "timestamp": start_time.isoformat(),
        }
        summary_path = os.path.join(args.output_dir, "cv_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Kết quả đã lưu: {summary_path}")

    print(f"⏱️  Tổng thời gian: {elapsed:.1f} phút")


if __name__ == "__main__":
    main()
