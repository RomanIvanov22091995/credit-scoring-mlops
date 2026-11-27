"""Prometheus-метрики приложения кредитного скоринга."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "scoring_api_requests_total",
    "Total HTTP requests", ["method", "endpoint", "status"])

REQUEST_LATENCY = Histogram(
    "scoring_api_request_latency_seconds",
    "HTTP request latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))

INFERENCE_LATENCY = Histogram(
    "scoring_model_inference_latency_seconds",
    "ONNX model inference latency", ["model_version"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5))

PREDICTIONS_TOTAL = Counter(
    "scoring_model_predictions_total",
    "Total scored applications", ["model_version"])

PREDICTED_PROBABILITY = Histogram(
    "scoring_model_predicted_probability",
    "Distribution of predicted default probability", ["model_version"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95))

CACHE_OPERATIONS = Counter(
    "scoring_api_cache_operations_total",
    "Redis cache operations", ["result"])

MODEL_INFO = Gauge(
    "scoring_model_info",
    "Model metadata (1 = текущая версия)", ["model_version", "model_file", "framework"])

DRIFT_SHARE = Gauge(
    "scoring_drift_share_drifted_features",
    "Share of drifted features (последний Evidently-отчёт)")

DRIFT_AUC = Gauge(
    "scoring_drift_test_auc",
    "AUC на свежих данных (последний Evidently-отчёт)")

DRIFT_TIMESTAMP = Gauge(
    "scoring_drift_last_check_timestamp_seconds",
    "Unix-время последней проверки дрифта")

HEALTH_GAUGE = Gauge(
    "scoring_api_health",
    "1 = модель загружена и готова")


def set_model_info(version: str, model_file: str, framework: str) -> None:
    MODEL_INFO.labels(version, model_file, framework).set(1)
