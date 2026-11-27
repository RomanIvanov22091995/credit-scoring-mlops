"""Конвертация PyTorch-модели в ONNX и сравнение производительности инференса.

Этапы:
  1. Загрузка models/trained/model.pt.
  2. Экспорт в ONNX (opset 15, динамический batch).
  3. Проверка модели onnx.checker.
  4. Сравнение латентности инференса PyTorch vs ONNX Runtime на CPU
     (single sample и batch), сохранение отчёта.

Запуск:
    python scripts/model_training/onnx_conversion.py --config config/params.yaml
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
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import CreditMLP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("onnx_conversion")


def inline_external_data(model_proto: onnx.ModelProto) -> None:
    """Новые версии torch.onnx выносят крупные тензоры в .data-файл.
    Для прод-деплоя нужен одиночный артефакт — инлайним веса обратно."""
    for tensor in model_proto.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]


def export_onnx(checkpoint_path: str, onnx_path: str, opset: int, n_features: int) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = CreditMLP(
        n_features=arch["n_features"], hidden_dims=arch["hidden_dims"],
        dropout=arch["dropout"], use_batch_norm=arch["use_batch_norm"],
        activation=arch["activation"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # обязательно: фиксируем BatchNorm/Dropout в inference-режим

    dummy = torch.randn(1, n_features, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["float_input"],
        output_names=["probability"],
        dynamic_axes={
            "float_input": {0: "batch_size"},
            "probability": {0: "batch_size"},
        },
    )
    proto = onnx.load(onnx_path)
    inline_external_data(proto)
    onnx.save(proto, onnx_path)
    # удаляем .data, оставшийся от экспортёра
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    log.info("ONNX сохранён: %s (%.1f КБ)", onnx_path, os.path.getsize(onnx_path) / 1024)


def bench(fn, data: np.ndarray, n_warmup: int = 20, n_runs: int = 200) -> dict:
    """Замер латентности: N прогонов, перцентили в миллисекундах."""
    for _ in range(n_warmup):
        fn(data)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(data)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
        "p99_ms": round(times[int(len(times) * 0.99)], 3),
        "runs": n_runs,
    }


def main(cfg: dict) -> None:
    trained = cfg["paths"]["trained_models"]
    prod = cfg["paths"]["production_models"]
    os.makedirs(prod, exist_ok=True)

    ckpt_path = os.path.join(trained, "model.pt")
    onnx_path = os.path.join(prod, "model.onnx")

    export_onnx(ckpt_path, onnx_path, cfg["onnx"]["opset"], n_features=20)

    # Валидация графа onnx.checker
    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    log.info("onnx.checker: OK | opset=%d | узлов=%d",
             model_proto.opset_import[0].version, len(model_proto.graph.node))

    # ------- Сравнение производительности на CPU -------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    pt_model = CreditMLP(n_features=arch["n_features"], hidden_dims=arch["hidden_dims"],
                         dropout=arch["dropout"], use_batch_norm=arch["use_batch_norm"],
                         activation=arch["activation"])
    pt_model.load_state_dict(ckpt["state_dict"])
    pt_model.eval()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    results = {}
    for batch in (1, 32, 128):
        x = rng.normal(size=(batch, 20)).astype(np.float32)
        results[f"pytorch_batch_{batch}"] = bench(lambda d: pt_model(torch.from_numpy(d)), x)
        results[f"onnx_batch_{batch}"] = bench(lambda d: sess.run(None, {input_name: d}), x)

    speedup_single = (results["pytorch_batch_1"]["p50_ms"] /
                      max(results["onnx_batch_1"]["p50_ms"], 1e-9))
    log.info("PyTorch  batch=1  p50=%.3f ms", results["pytorch_batch_1"]["p50_ms"])
    log.info("ONNX     batch=1  p50=%.3f ms (ускорение x%.2f)",
             results["onnx_batch_1"]["p50_ms"], speedup_single)
    log.info("PyTorch  batch=32 p50=%.3f ms", results["pytorch_batch_32"]["p50_ms"])
    log.info("ONNX     batch=32 p50=%.3f ms", results["onnx_batch_32"]["p50_ms"])

    report = {
        "onnx_file": onnx_path,
        "size_kb": round(os.path.getsize(onnx_path) / 1024, 1),
        "opset": model_proto.opset_import[0].version,
        "providers": sess.get_providers(),
        "benchmark_cpu": results,
        "speedup_single_sample": round(speedup_single, 2),
    }
    with open(os.path.join(prod, "conversion_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Отчёт конвертации: %s", os.path.join(prod, "conversion_report.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        main(yaml.safe_load(fh))
