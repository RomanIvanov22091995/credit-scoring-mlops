"""Credit Scoring API — промышленный сервис инференса ONNX-модели.

Эндпоинты:
  POST /predict          — скоринг заявок (batch до 1000), опционально Redis-кеш
  GET  /health           — liveness
  GET  /ready            — readiness (модель загружена, тестовый инференс прошёл)
  GET  /version          — версия модели и метрики качества
  GET  /drift-status     — последний Evidently-отчёт (drift share, AUC)
  GET  /metrics          — Prometheus

Запуск (dev):  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import monitoring as mon
from api.model_wrapper import FEATURE_NAMES, model
from api.schemas import (HealthResponse, PredictionRequest, PredictionResponse,
                         PredictionResult, VersionResponse)

logging.basicConfig(level=logging.INFO,
                    format='{"time":"%(asctime)s","level":"%(levelname)s",'
                           '"logger":"%(name)s","msg":"%(message)s"}')
log = logging.getLogger("api.main")

DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev-local")

app = FastAPI(
    title="Credit Scoring API",
    description="Промышленный сервис скоринга: MLP -> ONNX Runtime",
    version=MODEL_VERSION,
)


@app.on_event("startup")
def startup() -> None:
    model.load()
    test_meta = model.meta.get("test", {})
    mon.set_model_info(MODEL_VERSION, os.getenv("MODEL_PATH", "model.onnx"),
                       model.meta.get("framework", "onnxruntime"))
    mon.HEALTH_GAUGE.set(1)
    drift_path = os.getenv("DRIFT_REPORT", "reports/drift/latest_drift.json")
    if os.path.exists(drift_path):
        with open(drift_path, encoding="utf-8") as fh:
            drift = json.load(fh)
        mon.DRIFT_SHARE.set(drift.get("share_drifted_features", 0))
        mon.DRIFT_AUC.set(drift.get("current_auc", 0) or 0)
        mon.DRIFT_TIMESTAMP.set(drift.get("timestamp", 0))
    log.info("API готов: версия=%s, AUC(test)=%s", MODEL_VERSION, test_meta.get("roc_auc"))


@app.middleware("http")
async def track_latency(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - t0
    mon.REQUESTS_TOTAL.labels(request.method, request.url.path,
                              response.status_code).inc()
    mon.REQUEST_LATENCY.labels(request.url.path).observe(latency)
    return response


def _to_row(app_item) -> dict:
    return {field: getattr(app_item, field) for field in FEATURE_NAMES}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not model.ready:
        raise HTTPException(status_code=503, detail="Модель ещё загружается")
    rows = [_to_row(a) for a in payload.applications]
    t0 = time.perf_counter()
    proba, cached = model.cached_or_predict(rows)
    infer_ms = (time.perf_counter() - t0) * 1000
    mon.INFERENCE_LATENCY.labels(MODEL_VERSION).observe(infer_ms / 1000)
    mon.PREDICTIONS_TOTAL.labels(MODEL_VERSION).inc(len(proba))
    for p in proba:
        mon.PREDICTED_PROBABILITY.labels(MODEL_VERSION).observe(p)

    preds = []
    for i, p in enumerate(proba):
        band = ("A (низкий риск)" if p < 0.2 else
                "B (умеренный)" if p < 0.4 else
                "C (повышенный)" if p < 0.6 else "D (высокий риск)")
        preds.append(PredictionResult(
            index=i, default_probability=round(p, 5),
            decision="APPROVE" if p < DECISION_THRESHOLD else "REJECT",
            score_band=band))
    return PredictionResponse(model_version=MODEL_VERSION, predictions=preds,
                              cached=cached, latency_ms=round(infer_ms, 3))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_ready=model.ready,
                          redis_connected=model.redis is not None)


@app.get("/ready")
def ready() -> dict:
    if not model.ready:
        raise HTTPException(status_code=503, detail="loading")
    try:
        model.health_inference()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"inference failed: {exc}") from exc
    return {"status": "ready"}


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(model_version=MODEL_VERSION,
                           model_file=os.getenv("MODEL_PATH", "model.onnx"),
                           framework=model.meta.get("framework", "onnxruntime"),
                           test_roc_auc=model.meta.get("test", {}).get("roc_auc"),
                           features=len(FEATURE_NAMES))


@app.get("/drift-status")
def drift_status() -> dict:
    drift_path = os.getenv("DRIFT_REPORT", "reports/drift/latest_drift.json")
    if not os.path.exists(drift_path):
        return {"status": "no_data", "message": "Evidently-отчёт не найден"}
    with open(drift_path, encoding="utf-8") as fh:
        data = json.load(fh)
    # порог согласован с алертом ScoringDataDriftDetected и Airflow-триггером
    data["status"] = ("drift_detected" if data.get("share_drifted_features", 0) >= 0.25
                      else "stable")
    return data


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
