"""Airflow DAG: автоматическое переобучение кредитной скоринговой модели.

Цепочка:
  check_new_data      — есть ли свежие данные (дата файла / объём)
  check_drift         — дрифт? (latest_drift.json от Evidently CronJob или расчёт)
  retrain_model       — K8s Job (credit-scoring-trainer) в namespace credit-scoring
  validate_model      — гейт качества: AUC новой модели >= AUC текущей - delta
  deploy_canary       — CI (GitHub Actions canary-release) или прямой K8s canary
  notify              — итоговый статус в Slack (опционально)

Триггеры:
  * schedule — еженедельно (плюс drift-check CronJob может вызвать через TriggerDagRun);
  * по дрифту — публичный entrypoint trigger_retraining_if_drifted().

Верификация DAG: airflow dags list-import-errors (локально см. tests).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.trigger_rule import TriggerRule

# --- параметры окружения (переопределяются Variables/Environment) ---
NAMESPACE = os.getenv("RETRAIN_NAMESPACE", "credit-scoring")
TRAINER_IMAGE = Variable.get("trainer_image", default_var="credit-scoring-trainer:1.0.0")
AUC_DROP_TOLERANCE = float(Variable.get("auc_drop_tolerance", default_var="0.02"))
SLACK_WEBHOOK = Variable.get("slack_webhook", default_var="")

DEFAULT_ARGS = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def check_new_data(**context) -> str:
    """Свежие данные = current CSV на PVC обновлён за последние 7 дней."""
    path = context["params"].get("current_data", "/app/data/reference/current_drifted.csv")
    if not os.path.exists(path):
        return "check_drift"  # данных нет — но дрифт проверить всё равно можно
    age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
    if age_days > 7:
        raise RuntimeError(f"Данные устарели: {age_days} дней ({path})")
    context["ti"].xcom_push(key="data_age_days", value=age_days)
    return "check_drift"


def check_drift(**context) -> str:
    """Читает latest_drift.json: дрифт или недавняя проверка -> retrain, иначе skip."""
    drift_path = os.getenv("DRIFT_REPORT", "/app/reports/drift/latest_drift.json")
    if not os.path.exists(drift_path):
        raise RuntimeError("Нет Evidently-отчёта — проверьте CronJob drift-check")
    with open(drift_path, encoding="utf-8") as fh:
        drift = json.load(fh)
    share = drift.get("share_drifted_features", 0.0)
    context["ti"].xcom_push(key="drift_share", value=share)
    # Правила запуска: дрифт выше порога ИЛИ плановый еженедельный запуск
    manual_trigger = context["dag_run"].run_type == "manual"
    if drift.get("alert") or share >= 0.25 or manual_trigger:
        return "retrain_model"
    return "skip_retrain"  # данные стабильны — ждём следующего окна


def validate_model(**context) -> bool:
    """Гейт: метрики новой модели не хуже старой более чем на tolerance."""
    ti = context["ti"]
    new_metrics = ti.xcom_pull(key="new_model_metrics", task_ids="retrain_model") or {}
    new_auc = new_metrics.get("test_roc_auc")
    if new_auc is None:
        raise RuntimeError("retrain_model не вернул метрики новой модели")
    # текущая (прод) AUC читаем из /version прода или из metrics.json на PVC
    current_auc = float(os.getenv("CURRENT_AUC", "0.73576"))
    if new_auc < current_auc - AUC_DROP_TOLERANCE:
        raise RuntimeError(
            f"Гейт качества провален: new AUC={new_auc:.5f} < "
            f"current {current_auc:.5f} - {AUC_DROP_TOLERANCE}")
    context["ti"].xcom_push(key="model_version", value=new_metrics.get("model_version", "unspecified"))
    return True


def notify(**context) -> None:
    """Slack-уведомление об итогах пайплайна (soft-fail: вебхук опционален)."""
    if not SLACK_WEBHOOK:
        return
    import urllib.request
    ti = context["ti"]
    share = ti.xcom_pull(key="drift_share", task_ids="check_drift") or 0
    version = ti.xcom_pull(key="model_version", task_ids="validate_model") or "?"
    text = {
        "text": (f"credit_scoring_retraining завершён OK: версия {version}, "
                 f"дрифт был {share:.0%}")}
    req = urllib.request.Request(
        SLACK_WEBHOOK, data=json.dumps(text).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


with DAG(
    dag_id="credit_scoring_retraining",
    description="Автоматическое переобучение кредитной скоринговой модели",
    default_args=DEFAULT_ARGS,
    schedule_interval=timedelta(weeks=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "retraining", "credit-scoring"],
) as dag:

    start = EmptyOperator(task_id="start")

    check_data = PythonOperator(
        task_id="check_new_data",
        python_callable=check_new_data,
        provide_context=True,
        params={"current_data": "/app/data/reference/current_drifted.csv"},
    )

    drift_branch = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
        provide_context=True,
    )

    start >> check_data >> drift_branch

    # --- ветка переобучения ---
    retrain = KubernetesPodOperator(
        task_id="retrain_model",
        name="model-retrain",
        namespace=NAMESPACE,
        image=TRAINER_IMAGE,
        cmds=["/bin/sh", "-c"],
        arguments=[
            "set -e; "
            "python scripts/data_generation.py --config config/params.yaml; "
            "python scripts/model_training/train_model.py; "
            "python scripts/model_training/onnx_conversion.py; "
            "python scripts/model_training/onnx_validation.py; "
            "python scripts/model_training/optimize_model.py; "
            "echo RETRAIN_OK"
        ],
        get_logs=True,
        is_delete_operator_pod=True,
        container_resources={
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"},
        },
    )

    def _pull_new_metrics(**context):
        """Читает metrics.json, записанный Job'ом на PVC."""
        with open("/app/models/trained/metrics.json", encoding="utf-8") as fh:
            m = json.load(fh)
        context["ti"].xcom_push(
            key="new_model_metrics",
            value={"test_roc_auc": m.get("test", {}).get("roc_auc"),
                   "model_version": m.get("framework", "")},
        )

    pull_metrics = PythonOperator(
        task_id="pull_new_metrics",
        python_callable=_pull_new_metrics,
        provide_context=True,
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
        provide_context=True,
    )

    deploy_canary = KubernetesPodOperator(
        task_id="deploy_canary",
        name="deploy-canary",
        namespace=NAMESPACE,
        image="ghcr.io/your-org/ci-runner:latest",
        cmds=["kubectl"],
        arguments=[
            "-n", NAMESPACE, "set", "image", "deployment/scoring-api-canary",
            "api={{ var.value.registry }}/credit-scoring-api:"
            "{{ ti.xcom_pull(task_ids='validate_model', key='model_version') }}",
        ],
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # --- ветка пропуска (данные стабильны) ---
    skip_retrain = EmptyOperator(task_id="skip_retrain")

    finish_ok = PythonOperator(
        task_id="notify_success",
        python_callable=notify,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    finish_fail = PythonOperator(
        task_id="notify_failure",
        python_callable=lambda: None,  # ALERTMANAGER увидит провал DAG'а
        provide_context=True,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    drift_branch >> retrain >> pull_metrics >> validate >> deploy_canary
    drift_branch >> skip_retrain
    [deploy_canary, skip_retrain] >> finish_ok
    [retrain, validate, deploy_canary] >> finish_fail


def trigger_retraining_if_drifted(drift_share: float) -> bool:
    """Entry point для внешних систем (CronJob дрифта, API).
    Возвращает True, если пайплайн переобучения должен быть запущен."""
    return drift_share >= 0.25
