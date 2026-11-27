"""Валидация корректности конвертации PyTorch -> ONNX.

Проверки (все обязательны для выката модели в production):
  1. Структурная валидация: onnx.checker, opset, имена входов/выходов,
     динамический batch.
  2. Численная эквивалентность: ONNX Runtime vs PyTorch на:
     - случайных данных (N=5000);
     - реальном test-датасете;
     - граничных случаях (нули, экстремальные значения);
  3. Согласованность предсказаний классов (порог 0.5) — 100%;
  4. Допуск по вероятностям: max |p_pt - p_onnx| <= tol.
Код возврата != 0 при любой ошибке — CI-гейт перед деплоем.

Запуск:
    python scripts/model_training/onnx_validation.py --tol 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import CreditMLP  # noqa: E402
from data_generation import FEATURE_NAMES, TARGET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("onnx_validation")


def load_pytorch(checkpoint_path: str) -> CreditMLP:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = CreditMLP(n_features=arch["n_features"], hidden_dims=arch["hidden_dims"],
                      dropout=arch["dropout"], use_batch_norm=arch["use_batch_norm"],
                      activation=arch["activation"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main(cfg: dict, tol: float) -> int:
    trained = cfg["paths"]["trained_models"]
    prod = cfg["paths"]["production_models"]
    onnx_path = os.path.join(prod, "model.onnx")

    checks: dict[str, bool] = {}
    failures: list[str] = []

    # ---------- 1. Структурная валидация ----------
    proto = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(proto)
        checks["onnx_checker"] = True
    except Exception as exc:  # noqa: BLE001
        checks["onnx_checker"] = False
        failures.append(f"onnx.checker: {exc}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    checks["input_name"] = inp.name == "float_input"
    checks["output_name"] = out.name == "probability"
    checks["input_shape_dynamic_batch"] = (inp.shape[0] in (None, "batch_size")
                                           and inp.shape[1] == proto.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    checks["input_dtype_float32"] = inp.type == "tensor(float)"
    if not all([checks["input_name"], checks["output_name"],
                checks["input_shape_dynamic_batch"], checks["input_dtype_float32"]]):
        failures.append(f"IO-сигнатуры: {checks}")

    # ---------- 2. Численная эквивалентность ----------
    model = load_pytorch(os.path.join(trained, "model.pt"))

    @torch.no_grad()
    def pt_proba(x: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(x.astype(np.float32))).numpy()

    def ort_proba(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {inp.name: x.astype(np.float32)})[0]

    rng = np.random.default_rng(123)
    cases: dict[str, np.ndarray] = {
        "random_5000": rng.normal(size=(5000, 20)).astype(np.float32),
        "zeros": np.zeros((8, 20), dtype=np.float32),
        "extremes": np.tile(np.array([18, 15000.0, 30000.0, 60, 1.0, 6, 9, 9, 0.0,
                                      1.0, 12, 0.0, 900000.0, 12, 0, 0, 0, 0, 1.0, 0],
                                     dtype=np.float32), (4, 1)),
    }
    test_csv = os.path.join(cfg["paths"]["processed_data"], "test.csv")
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        # Модель обучена на стандартизованных признаках — применяем тот же scaler
        import joblib
        scaler = joblib.load(os.path.join(trained, "scaler.pkl"))
        cases["test_dataset"] = scaler.transform(
            df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
        y_true = df[TARGET].values
    else:
        y_true = None

    report = {"tolerance": tol, "checks": checks, "datasets": {}}
    for name, x in cases.items():
        p_pt, p_ort = pt_proba(x), ort_proba(x)
        max_abs = float(np.max(np.abs(p_pt - p_ort)))
        mean_abs = float(np.mean(np.abs(p_pt - p_ort)))
        class_agree = float(np.mean((p_pt >= 0.5) == (p_ort >= 0.5)))
        ok = max_abs <= tol and class_agree == 1.0
        checks[f"parity_{name}"] = ok
        report["datasets"][name] = {
            "n": int(len(x)), "max_abs_diff": round(max_abs, 8),
            "mean_abs_diff": round(mean_abs, 8), "class_agreement": class_agree, "passed": ok,
        }
        if not ok:
            failures.append(f"parity_{name}: max_abs={max_abs:.2e} agree={class_agree}")
        log.info("%-14s n=%5d | max|Δp|=%.2e | mean|Δp|=%.2e | class agree=%.4f | %s",
                 name, len(x), max_abs, mean_abs, class_agree, "OK" if ok else "FAIL")

    # ROC-AUC ONNX на тесте должен совпадать с PyTorch-моделью
    if y_true is not None:
        from sklearn.metrics import roc_auc_score
        auc_ort = float(roc_auc_score(y_true, ort_proba(cases["test_dataset"])))
        auc_pt = float(roc_auc_score(y_true, pt_proba(cases["test_dataset"])))
        checks["auc_match_test"] = abs(auc_ort - auc_pt) <= 1e-6
        report["test_roc_auc_pytorch"] = round(auc_pt, 6)
        report["test_roc_auc_onnx"] = round(auc_ort, 6)
        if not checks["auc_match_test"]:
            failures.append(f"AUC mismatch: pt={auc_pt} onnx={auc_ort}")
        log.info("ROC-AUC test: pytorch=%.6f onnx=%.6f -> %s",
                 auc_pt, auc_ort, "OK" if checks["auc_match_test"] else "FAIL")

    report["all_passed"] = not failures
    out_path = os.path.join(prod, "validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Отчёт валидации: %s", out_path)

    if failures:
        for msg in failures:
            log.error("VALIDATION FAILED: %s", msg)
        return 1
    log.info("=== ВСЕ ПРОВЕРКИ КОНВЕРТАЦИИ ПРОЙДЕНЫ ===")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="максимально допустимое |Δ| вероятностей")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        sys.exit(main(yaml.safe_load(fh), args.tol))
