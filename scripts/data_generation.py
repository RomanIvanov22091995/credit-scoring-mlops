"""Генерация синтетического датасета кредитного скоринга.

Создаёт реалистичные данные о заёмщиках с 20 признаками и целевой
переменной `default`. Дополнительно формирует:
  * train / validation / test выборки (data/processed);
  * reference-выборку (эталон для Evidently, распределение = train);
  * current-выборку с намеренным data drift (для проверки мониторинга).

Запуск:
    python scripts/data_generation.py --config config/params.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("data_generation")

FEATURE_NAMES = [
    "age", "income_monthly", "loan_amount", "loan_term_months",
    "credit_utilization", "num_existing_loans", "delinquencies_24m",
    "late_payments_12m", "employment_years", "debt_to_income",
    "num_credit_inquiries", "credit_history_years", "revolving_balance",
    "num_credit_cards", "mortgage", "is_employed", "education_level",
    "marital_status", "region_risk_score", "phone_verified",
]
TARGET = "default"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_features(n: int, rng: np.random.Generator, severity: float = 0.0) -> pd.DataFrame:
    """Генерирует признаки. `severity` > 0 сдвигает распределения (data drift)."""
    age = np.clip(rng.normal(41 + 6 * severity, 12), 18, 75).astype(int)
    income = np.clip(rng.lognormal(mean=11.6 - 0.25 * severity, sigma=0.5), 15_000, 600_000)
    loan_amount = np.clip(rng.lognormal(mean=12.3 + 0.2 * severity, sigma=0.6), 30_000, 5_000_000)
    term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.10, 0.22, 0.35, 0.18, 0.15])
    utilization = np.clip(rng.beta(2.0, 3.0, size=n) + 0.15 * severity, 0, 1)
    existing_loans = rng.poisson(1.5 + severity, size=n)
    delinq = rng.poisson(0.4, size=n)
    late_payments = rng.poisson(0.8 + 0.5 * severity, size=n)
    employment_years = np.clip(rng.exponential(6, size=n), 0, 45)
    dti = np.clip(rng.beta(3.0, 6.0, size=n) + 0.1 * severity, 0, 1)
    inquiries = rng.poisson(1.2 + 0.8 * severity, size=n)
    history_years = np.clip(rng.exponential(9, size=n), 0, 40)
    revolving = (utilization * income * rng.uniform(0.5, 1.5, size=n)).round(2)
    cards = 1 + rng.poisson(1.3, size=n)
    mortgage = rng.binomial(1, 0.35 - 0.1 * severity, size=n)
    employed = rng.binomial(1, 0.88 - 0.15 * severity, size=n)
    education = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    marital = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.50, 0.10, 0.10])
    region_risk = np.clip(rng.normal(0.4 + 0.2 * severity, 0.15), 0, 1).round(3)
    phone = rng.binomial(1, 0.85, size=n)

    df = pd.DataFrame({
        "age": age, "income_monthly": income.round(2), "loan_amount": loan_amount.round(2),
        "loan_term_months": term, "credit_utilization": utilization.round(4),
        "num_existing_loans": existing_loans, "delinquencies_24m": delinq,
        "late_payments_12m": late_payments, "employment_years": employment_years.round(1),
        "debt_to_income": dti.round(4), "num_credit_inquiries": inquiries,
        "credit_history_years": history_years.round(1), "revolving_balance": revolving,
        "num_credit_cards": cards, "mortgage": mortgage, "is_employed": employed,
        "education_level": education, "marital_status": marital,
        "region_risk_score": region_risk, "phone_verified": phone,
    })
    return df[FEATURE_NAMES]


def add_target(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Целевая переменная — логистическая функция признаков + шум.

    Весам придан «экономический» смысл: риск повышают высокая долговая
    нагрузка, просрочки, низкий доход относительно кредита и т.п.
    """
    z = (
        0.000 - 2.0
        + 0.030 * (df["age"] < 25).astype(float)
        - 0.00002 * df["income_monthly"]
        + 0.00000012 * df["loan_amount"]
        + 0.010 * df["loan_term_months"]
        + 2.20 * df["credit_utilization"]
        + 0.15 * df["num_existing_loans"]
        + 0.55 * df["delinquencies_24m"]
        + 0.35 * df["late_payments_12m"]
        - 0.05 * df["employment_years"]
        + 3.00 * df["debt_to_income"]
        + 0.12 * df["num_credit_inquiries"]
        - 0.03 * df["credit_history_years"]
        + 0.0000015 * df["revolving_balance"]
        - 0.10 * df["mortgage"]
        - 1.20 * df["is_employed"]
        - 0.20 * df["education_level"]
        + 1.50 * df["region_risk_score"]
        - 0.40 * df["phone_verified"]
        + rng.normal(0, 0.8, size=len(df))
    )
    df = df.copy()
    df[TARGET] = rng.binomial(1, _sigmoid(z))
    return df


def drift_dataframe(df: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Добавляет к выборке дрифт отдельных признаков (ковариатный дрифт)."""
    out = df.copy()
    n = len(out)
    out["income_monthly"] = (out["income_monthly"] * (1 - 0.25 * severity) *
                             rng.normal(1.0, 0.05, n)).round(2)
    out["age"] = np.clip(out["age"] + int(4 * severity) + rng.normal(0, 1, n), 18, 80).astype(int)
    out["credit_utilization"] = np.clip(out["credit_utilization"] + 0.15 * severity, 0, 1)
    out["num_credit_inquiries"] = np.maximum(0, out["num_credit_inquiries"] + rng.poisson(2 * severity, n))
    return out


def main(cfg: dict, out_dirs: dict) -> None:
    rs = cfg["data"]["random_state"]
    rng = np.random.default_rng(rs)

    log.info("Генерация %d записей...", cfg["data"]["n_samples"])
    df = generate_features(cfg["data"]["n_samples"], rng)
    df = add_target(df, rng)
    rate = df[TARGET].mean()
    log.info("Доля дефолтов: %.3f", rate)

    # Перемешиваем и режем train/val/test
    df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)
    n_test = int(len(df) * cfg["data"]["test_size"])
    n_val = int(len(df) * cfg["data"]["val_size"])
    test_df = df.iloc[:n_test]
    val_df = df.iloc[n_test:n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    os.makedirs(out_dirs["raw"], exist_ok=True)
    os.makedirs(out_dirs["processed"], exist_ok=True)
    os.makedirs(out_dirs["reference"], exist_ok=True)

    train_df.to_csv(os.path.join(out_dirs["raw"], "credit_raw_full.csv"), index=False)
    train_df.to_csv(os.path.join(out_dirs["processed"], "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dirs["processed"], "validation.csv"), index=False)
    test_df.to_csv(os.path.join(out_dirs["processed"], "test.csv"), index=False)

    # Reference = отобранная часть train (эталон Evidently)
    ref_idx = train_df.sample(n=cfg["data"]["n_drift_samples"], random_state=rs).index
    ref = train_df.loc[ref_idx]
    ref.to_csv(os.path.join(out_dirs["reference"], "reference.csv"), index=False)

    # Current = дрифтованные данные из того же генератора
    rng_cur = np.random.default_rng(rs + 1)
    cur = generate_features(cfg["data"]["n_drift_samples"], rng_cur, severity=cfg["data"]["drift_severity"])
    cur = add_target(cur, rng_cur)
    cur = drift_dataframe(cur, cfg["data"]["drift_severity"], rng_cur)
    cur.to_csv(os.path.join(out_dirs["reference"], "current_drifted.csv"), index=False)
    # Стабильная контрольная выборка: held-out часть train (те же распределения)
    stable = train_df.drop(index=ref_idx).sample(n=cfg["data"]["n_drift_samples"],
                                                 random_state=rs + 3)
    stable.to_csv(os.path.join(out_dirs["reference"], "current_stable.csv"), index=False)

    meta = {
        "feature_names": FEATURE_NAMES,
        "target": TARGET,
        "n_train": len(train_df), "n_validation": len(val_df), "n_test": len(test_df),
        "default_rate_train": round(float(train_df[TARGET].mean()), 4),
        "default_rate_test": round(float(test_df[TARGET].mean()), 4),
    }
    with open(os.path.join(out_dirs["processed"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Готово: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/params.yaml")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = args.out_dir
    out_dirs = {
        "raw": os.path.join(base, cfg["paths"]["raw_data"]),
        "processed": os.path.join(base, cfg["paths"]["processed_data"]),
        "reference": os.path.join(base, cfg["paths"]["reference_data"]),
    }
    main(cfg, out_dirs)
