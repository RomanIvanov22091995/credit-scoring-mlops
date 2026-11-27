"""Pydantic-схемы API кредитного скоринга."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CreditApplication(BaseModel):
    """Заявка на кредит: 20 признаков, порядок задаётся feature_names.json."""

    age: int = Field(ge=18, le=100, description="Возраст заёмщика")
    income_monthly: float = Field(gt=0, description="Месячный доход")
    loan_amount: float = Field(gt=0, description="Сумма кредита")
    loan_term_months: int = Field(ge=3, le=120, description="Срок в месяцах")
    credit_utilization: float = Field(ge=0, le=1, description="Использование кредитных линий")
    num_existing_loans: int = Field(ge=0, le=30, description="Действующие кредиты")
    delinquencies_24m: int = Field(ge=0, le=50, description="Делинквентности за 24 мес.")
    late_payments_12m: int = Field(ge=0, le=50, description="Просрочки за 12 мес.")
    employment_years: float = Field(ge=0, le=70, description="Стаж работы")
    debt_to_income: float = Field(ge=0, le=1, description="Долг / доход")
    num_credit_inquiries: int = Field(ge=0, le=30, description="Запросы в БКИ")
    credit_history_years: float = Field(ge=0, le=70, description="Возраст кредитной истории")
    revolving_balance: float = Field(ge=0, description="Возобновляемый баланс")
    num_credit_cards: int = Field(ge=0, le=30, description="Кредитные карты")
    mortgage: int = Field(ge=0, le=1, description="Ипотека")
    is_employed: int = Field(ge=0, le=1, description="Занятость")
    education_level: int = Field(ge=0, le=4, description="Уровень образования (0-4)")
    marital_status: int = Field(ge=0, le=3, description="Семейное положение (0-3)")
    region_risk_score: float = Field(ge=0, le=1, description="Риск региона")
    phone_verified: int = Field(ge=0, le=1, description="Телефон подтверждён")

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35, "income_monthly": 120000, "loan_amount": 500000,
            "loan_term_months": 36, "credit_utilization": 0.35,
            "num_existing_loans": 2, "delinquencies_24m": 0, "late_payments_12m": 1,
            "employment_years": 6.5, "debt_to_income": 0.28,
            "num_credit_inquiries": 2, "credit_history_years": 10.0,
            "revolving_balance": 45000, "num_credit_cards": 3, "mortgage": 0,
            "is_employed": 1, "education_level": 2, "marital_status": 1,
            "region_risk_score": 0.4, "phone_verified": 1,
        }}}


class PredictionRequest(BaseModel):
    applications: list[CreditApplication] = Field(min_length=1, max_length=1000,
                                                  description="Список заявок (до 1000)")


class PredictionResult(BaseModel):
    index: int
    default_probability: float = Field(ge=0, le=1)
    decision: str
    score_band: str


class PredictionResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResult]
    cached: bool = False
    latency_ms: float


class VersionResponse(BaseModel):
    model_version: str
    model_file: str
    framework: str
    test_roc_auc: float | None = None
    features: int


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    redis_connected: bool
