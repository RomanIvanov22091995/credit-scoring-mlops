"""Оптимизация модели: INT8-квантизация (ONNX Runtime) + structured pruning.

1. Динамическая INT8-квантизация model.onnx -> model_int8.onnx
   (квантизация весов MatMul/Gemm/Conv, активации в рантайме).
2. L1-unstructured pruning PyTorch-модели (30% весов в Linear-слоях)
   с коротким дообучением -> экспорт в ONNX (эксперимент).
3. Сравнение: размер файла, латентность, ROC-AUC на test.
Результат: models/production/optimization_report.json

Запуск:
    python scripts/model_training/optimize_model.py --config config/params.yaml
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
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn.utils.prune as prune
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import CreditMLP  # noqa: E402
from data_generation import FEATURE_NAMES, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("optimize_model")


def ort_session(path: str, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench_fn(fn, n_warmup: int = 30, n_runs: int = 300) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"p50_ms": round(times[len(times) // 2], 4),
            "p95_ms": round(times[int(len(times) * 0.95)], 4),
            "p99_ms": round(times[int(len(times) * 0.99)], 4),
            "throughput_per_sec": round(1000.0 / times[len(times) // 2], 0)}


def quantize(onnx_path: str, int8_path: str) -> None:
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,          # точнее, чем per-tensor для линейных слоёв
        reduce_range=False,
    )


def prune_and_export(ckpt: dict, amount: float, out_path: str, epochs: int = 2) -> dict:
    """L1-unstructured pruning 30% + короткое дообучение (восстановление качества)."""
    arch = ckpt["arch"]
    model = CreditMLP(n_features=arch["n_features"], hidden_dims=arch["hidden_dims"],
                      dropout=arch["dropout"], use_batch_norm=arch["use_batch_norm"],
                      activation=arch["activation"])
    model.load_state_dict(ckpt["state_dict"])

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    zeros = sum(int(torch.sum(m.weight_mask == 0)) for m in model.modules()
                if isinstance(m, torch.nn.Linear))
    total = sum(m.weight_mask.numel() for m in model.modules()
                if isinstance(m, torch.nn.Linear))
    sparsity = zeros / total
    log.info("Pruning: разреженность=%.1f%% (обнулено %d из %d весов)",
             sparsity * 100, zeros, total)

    # Короткое дообучение на маскированных весах (reparameterization)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
    import joblib
    scaler = joblib.load(os.path.join("models", "trained", "scaler.pkl"))
    X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    criterion = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        opt.step()
    # Финализация масок: удаляем reparameterization, оставляя разреженные веса
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, "weight")
    model.eval()

    dummy = torch.randn(1, arch["n_features"])
    torch.onnx.export(model, dummy, out_path, export_params=True, opset_version=15,
                      do_constant_folding=True, input_names=["float_input"],
                      output_names=["probability"],
                      dynamic_axes={"float_input": {0: "batch_size"},
                                    "probability": {0: "batch_size"}})
    # Инлайним external data, если экспортёр вынес веса в отдельный файл
    proto = onnx.load(out_path)
    for tensor in proto.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
    onnx.save(proto, out_path)
    if os.path.exists(out_path + ".data"):
        os.remove(out_path + ".data")
    return {"sparsity": round(sparsity, 4), "finetune_loss": round(float(loss.detach()), 5)}


def main(cfg: dict) -> None:
    prod = cfg["paths"]["production_models"]
    trained = cfg["paths"]["trained_models"]
    onnx_path = os.path.join(prod, "model.onnx")
    int8_path = os.path.join(prod, "model_int8.onnx")
    pruned_path = os.path.join(prod, "model_pruned.onnx")

    # ---------- Квантизация ----------
    log.info("=== INT8 динамическая квантизация ===")
    quantize(onnx_path, int8_path)
    size_fp32 = os.path.getsize(onnx_path) / 1024
    size_int8 = os.path.getsize(int8_path) / 1024
    log.info("Размер: fp32=%.1f КБ -> int8=%.1f КБ (сжатие x%.2f)",
             size_fp32, size_int8, size_fp32 / size_int8)

    # ---------- Pruning ----------
    log.info("=== L1-unstructured pruning (%.0f%%) ===", cfg["optimization"]["pruning_amount"] * 100)
    ckpt = torch.load(os.path.join(trained, "model.pt"), map_location="cpu", weights_only=False)
    prune_info = prune_and_export(ckpt, cfg["optimization"]["pruning_amount"], pruned_path)
    size_pruned = os.path.getsize(pruned_path) / 1024

    # ---------- Оценка качества на test ----------
    df = pd.read_csv(os.path.join(cfg["paths"]["processed_data"], "test.csv"))
    import joblib
    scaler = joblib.load(os.path.join(trained, "scaler.pkl"))
    X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
    y = df[TARGET].values

    def auc_of(path_or_model: str | CreditMLP) -> float:
        if isinstance(path_or_model, str):
            sess = ort_session(path_or_model)
            proba = sess.run(None, {sess.get_inputs()[0].name: X})[0]
        else:
            with torch.no_grad():
                proba = path_or_model(torch.from_numpy(X)).numpy()
        return float(roc_auc_score(y, proba))

    auc_fp32 = auc_of(onnx_path)
    auc_int8 = auc_of(int8_path)
    auc_pruned = auc_of(pruned_path)
    log.info("ROC-AUC: fp32=%.5f | int8=%.5f (Δ=%.2e) | pruned=%.5f",
             auc_fp32, auc_int8, auc_int8 - auc_fp32, auc_pruned)

    # ---------- Латентность (1 поток CPU, batch=1) ----------
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=(1, 20)).astype(np.float32)
    x32 = rng.normal(size=(32, 20)).astype(np.float32)

    def lat(sess, xb):
        name = sess.get_inputs()[0].name
        return bench_fn(lambda: sess.run(None, {name: xb}))

    sess_fp32, sess_int8, sess_pruned = (ort_session(onnx_path), ort_session(int8_path),
                                         ort_session(pruned_path))
    variants = {
        "onnx_fp32": {"latency_b1": lat(sess_fp32, x1), "latency_b32": lat(sess_fp32, x32),
                      "size_kb": round(size_fp32, 1), "roc_auc_test": round(auc_fp32, 5)},
        "onnx_int8_dynamic": {"latency_b1": lat(sess_int8, x1), "latency_b32": lat(sess_int8, x32),
                              "size_kb": round(size_int8, 1), "roc_auc_test": round(auc_int8, 5)},
        "onnx_pruned_30": {"latency_b1": lat(sess_pruned, x1), "latency_b32": lat(sess_pruned, x32),
                           "size_kb": round(size_pruned, 1), "roc_auc_test": round(auc_pruned, 5)},
    }
    for name, v in variants.items():
        log.info("%-18s b1 p50=%.3f ms | b32 p50=%.3f ms | %.1f КБ | AUC=%.5f",
                 name, v["latency_b1"]["p50_ms"], v["latency_b32"]["p50_ms"],
                 v["size_kb"], v["roc_auc_test"])

    report = {
        "variants": variants,
        "quantization": {"type": "dynamic_int8", "per_channel": True,
                         "size_reduction": round(size_fp32 / size_int8, 2)},
        "pruning": {"method": "l1_unstructured", **prune_info},
        "quality_threshold": {"max_auc_drop": 0.005,
                              "int8_ok": bool(abs(auc_int8 - auc_fp32) <= 0.005),
                              "pruned_ok": bool(abs(auc_pruned - auc_fp32) <= 0.01)},
    }
    with open(os.path.join(prod, "optimization_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Отчёт: %s", os.path.join(prod, "optimization_report.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        main(yaml.safe_load(fh))
