"""Locust-сценарий нагрузочного тестирования Credit Scoring API.

Запуск (web-UI):  locust -f scripts/model_training/load_test.py --host=http://localhost:8000
Запуск (headless): locust -f scripts/model_training/load_test.py --host=http://localhost:8000 \
                     --headless -u 50 -r 10 -t 60s --html reports/load_tests/load_report.html
"""
from __future__ import annotations

import json
import random

from locust import HttpUser, between, task

with open("config/feature_names.json", encoding="utf-8") as fh:
    FEATURES = json.load(fh)["feature_names"]

DEFAULTS = {
    "age": 38, "income_monthly": 130000, "loan_amount": 600000,
    "loan_term_months": 36, "credit_utilization": 0.4, "num_existing_loans": 2,
    "delinquencies_24m": 0, "late_payments_12m": 0, "employment_years": 7,
    "debt_to_income": 0.3, "num_credit_inquiries": 2, "credit_history_years": 11,
    "revolving_balance": 50000, "num_credit_cards": 3, "mortgage": 0,
    "is_employed": 1, "education_level": 2, "marital_status": 1,
    "region_risk_score": 0.4, "phone_verified": 1,
}
# Диапазоны для случайной генерации заявок (проверка устойчивости на вариативных данных)
RANGES = {
    "age": (18, 75), "income_monthly": (20000, 400000), "loan_amount": (30000, 3000000),
    "loan_term_months": (12, 60), "credit_utilization": (0, 1),
    "num_existing_loans": (0, 8), "delinquencies_24m": (0, 6), "late_payments_12m": (0, 8),
    "employment_years": (0, 40), "debt_to_income": (0, 0.9), "num_credit_inquiries": (0, 12),
    "credit_history_years": (0, 35), "revolving_balance": (0, 800000),
    "num_credit_cards": (1, 10), "region_risk_score": (0, 1),
}


def random_app() -> dict:
    row = {}
    for name in FEATURES:
        if name in RANGES:
            lo, hi = RANGES[name]
            v = random.uniform(lo, hi)
            row[name] = int(v) if name in ("age", "loan_term_months", "num_existing_loans",
                                           "delinquencies_24m", "late_payments_12m",
                                           "num_credit_inquiries", "num_credit_cards",
                                           "revolving_balance", "income_monthly",
                                           "loan_amount", "employment_years") else round(v, 4)
        else:
            row[name] = random.choice([0, 1])
    return row


class ScoringUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task(6)
    def predict_single(self):
        self.client.post("/predict", json={"applications": [DEFAULTS]},
                         name="POST /predict (fixed single)")

    @task(3)
    def predict_random_batch(self):
        batch = [random_app() for _ in range(random.choice([4, 16, 32]))]
        self.client.post("/predict", json={"applications": batch},
                         name="POST /predict (random batch)")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
