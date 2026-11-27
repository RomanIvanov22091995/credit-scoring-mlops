"""Тесты API: paritet с ONNX-валидацией, граничные случаи, контракты ответа.

Запуск: pytest tests/ -v
Требует: обученную модель (scripts/model_training/train_model.py) и ONNX.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MODEL_PATH", "models/production/model.onnx")

from api.main import app  # noqa: E402
from api.model_wrapper import FEATURE_NAMES  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """TestClient как контекст-менеджер: срабатывает startup (загрузка модели)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_app() -> dict:
    """Валидная заявка (средний клиент)."""
    return {
        "age": 35, "income_monthly": 120000, "loan_amount": 500000,
        "loan_term_months": 36, "credit_utilization": 0.35,
        "num_existing_loans": 2, "delinquencies_24m": 0, "late_payments_12m": 1,
        "employment_years": 6.5, "debt_to_income": 0.28, "num_credit_inquiries": 2,
        "credit_history_years": 10.0, "revolving_balance": 45000,
        "num_credit_cards": 3, "mortgage": 0, "is_employed": 1,
        "education_level": 2, "marital_status": 1, "region_risk_score": 0.4,
        "phone_verified": 1,
    }


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_ready"] is True

    def test_ready(self, client):
        r = client.get("/ready")
        assert r.status_code == 200

    def test_version_contract(self, client):
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert body["features"] == 20
        assert 0 < (body["test_roc_auc"] or 0) < 1


class TestPredict:
    def test_single_prediction_contract(self, client, sample_app):
        r = client.post("/predict", json={"applications": [sample_app]})
        assert r.status_code == 200
        body = r.json()
        p = body["predictions"][0]
        assert 0 <= p["default_probability"] <= 1
        assert p["decision"] in ("APPROVE", "REJECT")
        assert p["score_band"].startswith(("A", "B", "C", "D"))
        assert body["model_version"]
        assert body["latency_ms"] >= 0

    def test_batch(self, client, sample_app):
        r = client.post("/predict", json={"applications": [sample_app] * 32})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 32

    def test_risky_client_rejected(self, client):
        risky = {
            "age": 22, "income_monthly": 30000, "loan_amount": 2000000,
            "loan_term_months": 60, "credit_utilization": 0.95,
            "num_existing_loans": 7, "delinquencies_24m": 8, "late_payments_12m": 10,
            "employment_years": 0.5, "debt_to_income": 0.9, "num_credit_inquiries": 10,
            "credit_history_years": 0.5, "revolving_balance": 500000,
            "num_credit_cards": 6, "mortgage": 0, "is_employed": 0,
            "education_level": 0, "marital_status": 2, "region_risk_score": 0.95,
            "phone_verified": 0,
        }
        r = client.post("/predict", json={"applications": [risky]})
        p = r.json()["predictions"][0]
        assert p["decision"] == "REJECT"
        assert p["default_probability"] > 0.5

    def test_validation_error_422(self, client, sample_app):
        bad = dict(sample_app)
        bad["age"] = 10  # вне диапазона
        r = client.post("/predict", json={"applications": [bad]})
        assert r.status_code == 422

    def test_empty_batch_rejected(self, client):
        r = client.post("/predict", json={"applications": []})
        assert r.status_code == 422


class TestParityWithReference:
    """Прогон реального тест-датасета: API должен совпадать с ORT напрямую."""

    @pytest.fixture(scope="class")
    def test_data(self):
        if not os.path.exists("data/processed/test.csv"):
            pytest.skip("нет test.csv — запустите data_generation")
        return pd.read_csv("data/processed/test.csv").head(200)

    def test_api_matches_direct_ort(self, client, test_data):
        import onnxruntime as ort
        sess = ort.InferenceSession("models/production/model.onnx",
                                    providers=["CPUExecutionProvider"])
        scaler = joblib.load("models/trained/scaler.pkl")

        rows = test_data[FEATURE_NAMES].to_dict("records")
        # склеиваем партии <=1000
        resp = client.post("/predict", json={"applications": rows}).json()
        api_proba = np.array([p["default_probability"] for p in resp["predictions"]])

        X = scaler.transform(test_data[FEATURE_NAMES].values.astype(np.float32)).astype(np.float32)
        ort_proba = sess.run(None, {sess.get_inputs()[0].name: X})[0]

        assert np.max(np.abs(api_proba - ort_proba)) < 1e-5  # допуск на округление до 5 знаков

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        text = r.text
        for metric in ("scoring_api_requests_total",
                       "scoring_model_inference_latency_seconds",
                       "scoring_model_predictions_total"):
            assert metric in text
