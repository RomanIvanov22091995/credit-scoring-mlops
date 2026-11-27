"""Комплексное нагрузочное benchmark-тестирование модели на CPU.

Матрица эксперимента:
  * варианты: pytorch_fp32, onnx_fp32, onnx_int8, onnx_pruned;
  * batch: 1, 8, 32, 64, 128;
  * потоки ORT: 1, 2, 4, 8;
  * метрики: p50/p95/p99 латентность, throughput (req/s), ROC-AUC.

Результат: reports/benchmark/benchmark_results.json,
           reports/benchmark/benchmark_report.md,
           reports/benchmark/*.png (графики).

Запуск:
    python scripts/model_training/benchmark.py --config config/params.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import CreditMLP  # noqa: E402
from data_generation import FEATURE_NAMES, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark")

BATCH_SIZES = [1, 8, 32, 64, 128]
THREAD_COUNTS = [1, 2, 4, 8]


def make_session(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def measure(fn, n_warmup: int, n_runs: int) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": round(times[len(times) // 2], 4),
        "p95_ms": round(times[int(len(times) * 0.95)], 4),
        "p99_ms": round(times[int(len(times) * 0.99)], 4),
        "mean_ms": round(statistics.mean(times), 4),
        "throughput_rps": round(len(times) / (sum(times) / 1000.0), 0),
    }


def main(cfg: dict) -> None:
    prod, trained = cfg["paths"]["production_models"], cfg["paths"]["trained_models"]
    out_dir = os.path.join(cfg["paths"]["reports"], "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    variants: dict[str, dict] = {}
    ckpt = torch.load(os.path.join(trained, "model.pt"), map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    pt_model = CreditMLP(n_features=arch["n_features"], hidden_dims=arch["hidden_dims"],
                         dropout=arch["dropout"], use_batch_norm=arch["use_batch_norm"],
                         activation=arch["activation"])
    pt_model.load_state_dict(ckpt["state_dict"])
    pt_model.eval()
    variants["pytorch_fp32"] = {"type": "pytorch"}
    variants["onnx_fp32"] = {"type": "ort", "path": os.path.join(prod, "model.onnx")}
    variants["onnx_int8_dynamic"] = {"type": "ort", "path": os.path.join(prod, "model_int8.onnx")}
    variants["onnx_pruned_30"] = {"type": "ort", "path": os.path.join(prod, "model_pruned.onnx")}

    sessions = {name: make_session(v["path"], threads=1)
                for name, v in variants.items() if v["type"] == "ort"}

    # ---------- Качество (ROC-AUC на test) ----------
    df = pd.read_csv(os.path.join(cfg["paths"]["processed_data"], "test.csv"))
    import joblib
    scaler = joblib.load(os.path.join(trained, "scaler.pkl"))
    X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
    y = df[TARGET].values
    with torch.no_grad():
        auc_pt = roc_auc_score(y, pt_model(torch.from_numpy(X)).numpy())
    variants["pytorch_fp32"]["roc_auc_test"] = round(float(auc_pt), 5)
    variants["pytorch_fp32"]["size_kb"] = round(os.path.getsize(os.path.join(trained, "model.pt")) / 1024, 1)
    for name in sessions:
        proba = sessions[name].run(None, {sessions[name].get_inputs()[0].name: X})[0]
        variants[name]["roc_auc_test"] = round(float(roc_auc_score(y, proba)), 5)
        variants[name]["size_kb"] = round(os.path.getsize(variants[name]["path"]) / 1024, 1)
    log.info("AUC: %s", {k: v["roc_auc_test"] for k, v in variants.items()})

    # ---------- Латентность: варианты x batch ----------
    rng = np.random.default_rng(2024)
    lat_results: dict[str, dict[str, dict]] = {name: {} for name in variants}
    for name, meta in variants.items():
        for batch in BATCH_SIZES:
            x = rng.normal(size=(batch, 20)).astype(np.float32)
            n_warm = 30 if batch <= 32 else 10
            n_runs = 300 if batch <= 32 else 100
            if meta["type"] == "pytorch":
                fn = lambda d=x: pt_model(torch.from_numpy(d))  # noqa: E731
            else:
                s = sessions[name]
                inp = s.get_inputs()[0].name
                fn = lambda d=x: s.run(None, {inp: d})  # noqa: E731
            lat_results[name][str(batch)] = measure(fn, n_warm, n_runs)
        log.info("%-18s b1 p95=%.3f ms | b128 p95=%.3f ms",
                 name, lat_results[name]["1"]["p95_ms"], lat_results[name]["128"]["p95_ms"])

    # ---------- Масштабирование по потокам (ONNX fp32, batch=1) ----------
    thread_results: dict[str, dict] = {}
    for name in ("onnx_fp32", "onnx_int8_dynamic"):
        thread_results[name] = {}
        for threads in THREAD_COUNTS:
            s = make_session(variants[name]["path"], threads)
            inp = s.get_inputs()[0].name
            x = rng.normal(size=(1, 20)).astype(np.float32)
            thread_results[name][str(threads)] = measure(lambda: s.run(None, {inp: x}), 30, 300)
        log.info("%s потоки: %s", name,
                 {t: v["p50_ms"] for t, v in thread_results[name].items()})

    # ---------- Сохранение ----------
    payload = {
        "hardware": {"platform": sys.platform, "cpu": os.cpu_count(),
                     "note": "Apple Silicon (M-series), CPUExecutionProvider"},
        "batch_sizes": BATCH_SIZES, "thread_counts": THREAD_COUNTS,
        "variants": variants, "latency_by_batch": lat_results,
        "thread_scaling": thread_results,
    }
    with open(os.path.join(out_dir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---------- Графики ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name in lat_results:
            p50 = [lat_results[name][str(b)]["p50_ms"] for b in BATCH_SIZES]
            p95 = [lat_results[name][str(b)]["p95_ms"] for b in BATCH_SIZES]
            axes[0].plot(BATCH_SIZES, p50, marker="o", label=f"{name} p50")
            axes[1].plot(BATCH_SIZES, p95, marker="s", linestyle="--", label=f"{name} p95")
        for ax, title in zip(axes, ["Latency p50 vs batch", "Latency p95 vs batch"]):
            ax.set_xlabel("batch size"); ax.set_ylabel("ms")
            ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "latency_vs_batch.png"), dpi=130)

        fig2, ax = plt.subplots(figsize=(8, 5))
        names = list(variants)
        aucs = [variants[n]["roc_auc_test"] for n in names]
        sizes = [variants[n]["size_kb"] for n in names]
        ax.bar(names, sizes, color="#4C72B0")
        for i, (n, a) in enumerate(zip(names, aucs)):
            ax.text(i, sizes[i] + max(sizes) * 0.01, f"AUC {a}", ha="center", fontsize=9)
        ax.set_ylabel("размер, КБ"); ax.set_title("Размер модели и качество (test ROC-AUC)")
        ax.grid(alpha=0.3, axis="y")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "size_auc.png"), dpi=130)
        log.info("Графики сохранены в %s/", out_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Графики не построены: %s", exc)

    log.info("Результаты: %s/benchmark_results.json", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        main(yaml.safe_load(fh))
