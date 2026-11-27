"""Тесты финализации модели: артефакты, INT8-качество, отчёты Stage 1.

Запуск: pytest tests/test_model.py -v   (после полного пайплайна Stage 1)
"""
from __future__ import annotations

import json
import os

import pytest

PROD = "models/production"
TRAINED = "models/trained"


def _skip_if_no_artifacts():
    if not os.path.exists(os.path.join(PROD, "model.onnx")):
        pytest.skip("артефакты Stage 1 отсутствуют — запустите пайплайн")


class TestArtifacts:
    def test_required_files_exist(self):
        _skip_if_no_artifacts()
        for f in (os.path.join(PROD, "model.onnx"),
                  os.path.join(PROD, "model_int8.onnx"),
                  os.path.join(TRAINED, "model.pt"),
                  os.path.join(TRAINED, "scaler.pkl"),
                  os.path.join(TRAINED, "metrics.json")):
            assert os.path.exists(f), f"нет артефакта {f}"

    def test_model_quality_sane(self):
        _skip_if_no_artifacts()
        metrics = json.load(open(os.path.join(TRAINED, "metrics.json")))
        auc = metrics["test"]["roc_auc"]
        assert 0.6 < auc < 0.95, f"подозрительный AUC: {auc}"

    def test_quantization_preserves_quality(self):
        _skip_if_no_artifacts()
        opt = json.load(open(os.path.join(PROD, "optimization_report.json")))
        auc_fp32 = opt["variants"]["onnx_fp32"]["roc_auc_test"]
        auc_int8 = opt["variants"]["onnx_int8_dynamic"]["roc_auc_test"]
        assert abs(auc_int8 - auc_fp32) <= 0.005

    def test_validation_report_passed(self):
        if not os.path.exists(os.path.join(PROD, "validation_report.json")):
            pytest.skip("нет validation_report.json")
        report = json.load(open(os.path.join(PROD, "validation_report.json")))
        assert report["all_passed"] is True

    def test_quantization_compresses_or_matches(self):
        """INT8 должен сжимать модель на достаточно большой сети."""
        _skip_if_no_artifacts()
        opt = json.load(open(os.path.join(PROD, "optimization_report.json")))
        size_fp32 = opt["variants"]["onnx_fp32"]["size_kb"]
        size_int8 = opt["variants"]["onnx_int8_dynamic"]["size_kb"]
        assert size_int8 < size_fp32 * 0.6, "INT8 не дал сжатия"
