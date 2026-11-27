"""Юнит-тесты DAG Airflow без полного окружения (CI-гейт синтаксиса структуры)."""
from __future__ import annotations

import ast
import os
import sys

DAG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "airflow", "dags", "retraining_dag.py")


def test_dag_file_exists():
    assert os.path.exists(DAG_PATH)


def test_dag_compiles():
    src = open(DAG_PATH, encoding="utf-8").read()
    ast.parse(src)  # SyntaxError если сломан


def test_dag_structure():
    """DAG-файл должен содержать ключевые задачи и корректную цепочку."""
    src = open(DAG_PATH, encoding="utf-8").read()
    for task in ("check_new_data", "check_drift", "retrain_model",
                 "validate_model", "deploy_canary"):
        assert f'task_id="{task}"' in src, f"нет задачи {task}"
    # цепочка полного пайплайна
    assert "retrain >> pull_metrics >> validate >> deploy_canary" in src
    # ветвление: skip-ветка существует
    assert "skip_retrain" in src
    # триггер дрифта порогом
    assert ">= 0.25" in src


def test_dag_has_quality_gate():
    src = open(DAG_PATH, encoding="utf-8").read()
    assert "AUC_DROP_TOLERANCE" in src
    assert "Гейт качества провален" in src
