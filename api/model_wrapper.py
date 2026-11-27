"""Обёртка над ONNX-моделью: загрузка, препроцессинг, инференс, кеш Redis."""
from __future__ import annotations

import json
import logging
import os
import threading

import joblib
import numpy as np
import onnxruntime as ort

log = logging.getLogger("api.model")

MODEL_PATH = os.getenv("MODEL_PATH", "models/production/model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "models/trained/scaler.pkl")
META_PATH = os.getenv("META_PATH", "models/trained/metrics.json")
FEATURE_PATH = os.getenv("FEATURE_PATH", "config/feature_names.json")
ORT_THREADS = int(os.getenv("ORT_THREADS", "1"))
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))

FEATURE_NAMES: list[str] = []


class ModelWrapper:
    """Потокобезопасная обёртка: один InferenceSession, scaler, опциональный Redis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session: ort.InferenceSession | None = None
        self.scaler = None
        self.meta: dict = {}
        self.redis = None
        self.cache_hits = 0
        self.cache_misses = 0

    # ---------- lifecycle ----------
    def load(self) -> None:
        with self._lock:
            if self.session is not None:
                return
            so = ort.SessionOptions()
            so.intra_op_num_threads = ORT_THREADS
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                MODEL_PATH, so, providers=["CPUExecutionProvider"])
            self.scaler = joblib.load(SCALER_PATH)
            for path, key in ((META_PATH, "meta"), (FEATURE_PATH, "features")):
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                        if key == "meta":
                            self.meta = data
                        else:
                            FEATURE_NAMES.extend(data["feature_names"])
            if not FEATURE_NAMES:
                # fallback: порядок признаков из meta.json, записанного при обучении
                FEATURE_NAMES.extend(self.meta.get("feature_names", []))
            if REDIS_URL:
                try:
                    import redis
                    self.redis = redis.Redis.from_url(
                        REDIS_URL, socket_timeout=0.2, decode_responses=True)
                    self.redis.ping()
                    log.info("Redis-кеш подключён: %s", REDIS_URL)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Redis недоступен (%s) — работаем без кеша", exc)
                    self.redis = None
            log.info("Модель загружена: %s | версию метрик см. /version", MODEL_PATH)

    @property
    def ready(self) -> bool:
        return self.session is not None

    # ---------- inference ----------
    def health_inference(self) -> float:
        """Прогон одной записи для readiness-пробы."""
        x = self.scaler.transform(
            np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)).astype(np.float32)
        return float(self._run(x)[0])

    def _run(self, xs: np.ndarray) -> np.ndarray:
        assert self.session is not None
        return self.session.run(None, {self.session.get_inputs()[0].name: xs})[0]

    def predict_proba(self, rows: list[dict]) -> list[float]:
        X = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows],
                     dtype=np.float32)
        X = self.scaler.transform(X).astype(np.float32)
        return [float(p) for p in self._run(X)]

    # ---------- кеш ----------
    @staticmethod
    def _key(payload: list[dict]) -> str:
        import hashlib
        return "pred:" + hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def cached_or_predict(self, payload: list[dict]) -> tuple[list[float], bool]:
        if self.redis is None:
            return self.predict_proba(payload), False
        key = self._key(payload)
        try:
            hit = self.redis.get(key)
            if hit:
                self.cache_hits += 1
                return json.loads(hit), True
        except Exception:  # noqa: BLE001
            pass
        self.cache_misses += 1
        proba = self.predict_proba(payload)
        try:
            self.redis.setex(key, CACHE_TTL, json.dumps(proba))
        except Exception:  # noqa: BLE001
            pass
        return proba, False


model = ModelWrapper()
