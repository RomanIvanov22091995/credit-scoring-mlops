"""Мониторинг дрифта данных и качества модели (Evidently AI).

Что делает:
  1. Загружает reference (обучающая выборка) и current (свежие данные).
  2. DataDriftTable — дрифт каждого из 20 признаков.
  3. ClassificationQuality — деградация качества (AUC) на current с истиной.
  4. Сохраняет HTML-отчёт + latest_drift.json (его читает API /drift-status
     и CronJob → Prometheus / алертинг → Airflow-триггер переобучения).

Режимы current-данных (аргумент --current):
  * путь к CSV с колонкой default;
  * auto (по умолчанию): data/reference/current_drifted.csv,
    fallback current_stable.csv.

Запуск:
    python scripts/drift/drift_monitoring.py            # локально (дрифтованные)
    python scripts/drift/drift_monitoring.py --threshold 0.4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("drift_monitoring")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_generation import FEATURE_NAMES, TARGET  # noqa: E402

DEFAULT_THRESHOLD = 0.25  # доля дрифтованных признаков, после которой алерт


def run_drift(reference_path: str, current_path: str, threshold: float,
              out_dir: str, model_auc_current: float | None = None,
              model_path: str = "models/production/model.onnx",
              scaler_path: str = "models/trained/scaler.pkl") -> dict:
    """Строит Evidently-отчёты и возвращает машинночитаемую сводку."""
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset
    from evidently.metrics import DataDriftTable, DatasetDriftMetric

    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)
    ref = ref[FEATURE_NAMES + [TARGET]]
    cur = cur[FEATURE_NAMES + [TARGET]]

    # --- Data drift (PSI — стандарт скоринговых карт) ---
    drift_report = Report(metrics=[
        DatasetDriftMetric(stattest="psi", stattest_threshold=0.2),
        DataDriftTable(stattest="psi", stattest_threshold=0.2),
    ])
    drift_report.run(reference_data=ref, current_data=cur)

    # --- Качество на current: prediction колонка от ONNX-модели ---
    if model_auc_current is not None:
        try:
            from evidently.pipeline.column_mapping import ColumnMapping
            import joblib
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            scaler = joblib.load(scaler_path)

            def add_prediction(df: pd.DataFrame) -> pd.DataFrame:
                X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
                proba = sess.run(None, {sess.get_inputs()[0].name: X})[0]
                out = df.copy()
                out["prediction"] = proba.astype(float)
                return out

            cm = ColumnMapping(target=TARGET, prediction="prediction",
                               numerical_features=FEATURE_NAMES, categorical_features=[])
            quality_report = Report(metrics=[ClassificationPreset()])
            quality_report.run(reference_data=add_prediction(ref),
                               current_data=add_prediction(cur),
                               column_mapping=cm)
            quality_report.save_html(os.path.join(out_dir, "quality_report.html"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Classification-отчёт не построен: %s", exc)
    drift_report.save_html(os.path.join(out_dir, "drift_report.html"))

    # --- Машинночитаемая сводка ---
    dataset_drifted = False
    dataset_share = 0.0
    drifted_features: list[str] = []
    drift_scores: dict[str, float] = {}
    for m in drift_report.as_dict().get("metrics", []):
        result = m.get("result", {})
        if "dataset_drift" in result:
            dataset_drifted = result["dataset_drift"]
            dataset_share = result.get("share_of_drifted_columns", 0.0)
        by_col = result.get("drift_by_columns", {})
        for col, info in by_col.items():
            drift_scores[col] = round(float(info.get("drift_score", 0.0)), 4)
            if info.get("drift_detected") in (True, "drift detected"):
                drifted_features.append(col)

    summary = {
        "timestamp": int(time.time()),
        "dataset_drift": bool(dataset_drifted),
        "share_drifted_features": round(float(dataset_share), 4),
        "n_features": len(FEATURE_NAMES),
        "drifted_features": drifted_features,
        "drift_scores_psi": drift_scores,
        "drift_threshold_share": threshold,
        "alert": bool(dataset_share >= threshold),
        "reference": reference_path,
        "current": current_path,
    }
    if model_auc_current is not None:
        summary["current_auc"] = round(float(model_auc_current), 5)
    return summary


def main(cfg: dict, current: str, threshold: float) -> int:
    ref_dir = cfg["paths"]["reference_data"]
    out_dir = os.path.join(cfg["paths"]["reports"], "drift")
    os.makedirs(out_dir, exist_ok=True)

    reference_path = os.path.join(ref_dir, "reference.csv")
    if not os.path.exists(current):
        # auto: сначала «дрифтованный» сценарий
        candidate = os.path.join(ref_dir, "current_drifted.csv")
        current = candidate if os.path.exists(candidate) else os.path.join(ref_dir, "current_stable.csv")
    log.info("Reference: %s | Current: %s", reference_path, current)

    # AUC на current через ONNX-модель (concept drift / performance decay)
    current_auc: float | None = None
    try:
        import joblib
        import numpy as np
        import onnxruntime as ort
        from sklearn.metrics import roc_auc_score

        model_path = os.path.join(cfg["paths"]["production_models"], "model.onnx")
        scaler_path = os.path.join(cfg["paths"]["trained_models"], "scaler.pkl")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            df = pd.read_csv(current)
            scaler = joblib.load(scaler_path)
            X = scaler.transform(df[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            proba = sess.run(None, {sess.get_inputs()[0].name: X})[0]
            current_auc = float(roc_auc_score(df[TARGET].values, proba))
            log.info("AUC на current: %.5f", current_auc)
    except Exception as exc:  # noqa: BLE001
        log.warning("AUC-check пропущен: %s", exc)

    summary = run_drift(reference_path, current, threshold, out_dir, current_auc,
                        model_path=os.path.join(cfg["paths"]["production_models"], "model.onnx"),
                        scaler_path=os.path.join(cfg["paths"]["trained_models"], "scaler.pkl"))

    with open(os.path.join(out_dir, "latest_drift.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Сводка: %s", json.dumps(summary, ensure_ascii=False))

    if summary["alert"]:
        log.error("DRIFT ALERT: %d%% признаков дрейфуют (порог %.0f%%) — "
                  "запускается триггер переобучения",
                  summary["share_drifted_features"] * 100, threshold * 100)
        return 2  # спец-код: алерт дрифта (Airflow/CronJob его интерпретируют)
    log.info("Дрифт не выявлен (или ниже порога)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--current", default="",
                        help="CSV с текущими данными (по умолчанию auto)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        sys.exit(main(yaml.safe_load(fh), args.current, args.threshold))
