# Архитектура: Credit Scoring System (промышленный MLOps)

## 1. Общая схема

```
                         ┌──────────────────────────────────────────────────────┐
                         │                    Yandex Cloud                       │
                         │                                                       │
 Developer ──push──▶ GitHub Actions ──▶ GHCR (образы)                           │
    │                    │                        │                              │
    │            build → test → security scan      │                              │
    │                    │                        ▼                              │
    │                    │              ┌───────────────────────┐                │
    │                    └──kubectl──▶ │  Managed Kubernetes   │                │
    │                                   │  ┌─────────────────┐  │                │
    ▼                                   │  │ nginx-ingress    │  │                │
 Object Storage (TF state,              │  │   ↓              │  │                │
 DVC remote, MLflow)                    │  │ frontend (nginx) │  │                │
                                        │  │   ↓ /api         │  │                │
 Client ──▶ LB/Ingress ───────────────▶ │  │ scoring-api ×N   │──┼──▶ Redis (кеш) │
                                        │  │   │ ONNX Runtime │  │                │
                                        │  └───┼──────────────┼──┘                │
                                        │      ▼              ▼                   │
                                        │  CronJob drift    HPA 3→12             │
                                        │  (Evidently PSI)   (CPU 65%)           │
                                        │      │                                 │
                                        │      ▼                                 │
                                        │  Airflow DAG: дрифт/расписание →       │
                                        │  Job retrain → гейт AUC → canary       │
                                        │                                        │
                                        │  monitoring ns: Prometheus, Grafana,   │
                                        │  Alertmanager, Loki + Promtail         │
                                        └────────────────────────────────────────┘
```

## 2. Поток данных (inference path)

1. **Клиент** (web-форма / интеграция) → HTTPS → Ingress nginx.
2. **Ingress** маршрутизирует: `/api/*` → Service `scoring-api:8000`, `/` → frontend.
3. **scoring-api** (FastAPI, 3+ реплик, HPA до 12):
   - валидация схемы заявки (pydantic, 20 признаков с диапазонами);
   - поиск в Redis-кеше (ключ = SHA1 payload, TTL 300 с) — при попадании ответ мгновенный;
   - препроцессинг: `StandardScaler` (тот же, что при обучении — артефакт `scaler.pkl`);
   - инференс: ONNX Runtime (INT8-модель, 1 поток на под, горизонтальное масштабирование);
   - решение: `APPROVE` при P(default) < 0.5 (настраивается), скоринговый бэнд A–D;
   - экспорт метрик Prometheus: латентности (histogram), счётчики, распределение вероятностей.
4. **Отказоустойчивость**: readiness-проба (тестовый инференс) исключает неготовый под из LB;
   PDB держит ≥2 реплики; rolling update без простоя; Redis — опционален (degraded mode).

## 3. Поток обучения (training path)

```
scripts/data_generation.py ─▶ data/processed/*.csv ─▶ DVC remote (Object Storage)
        │
        ▼
train_model.py (PyTorch MLP, early stopping по val ROC-AUC)
        │  artifacts: model.pt + scaler.pkl + metrics.json
        ▼
onnx_conversion.py ─▶ model.onnx (opset 18, dynamic batch)
        │
onnx_validation.py ─▶ ГЕЙТ: |Δp| ≤ 1e-4, 100% согласованность классов, AUC match
        │
optimize_model.py ─▶ model_int8.onnx (×3.24 сжатие) + model_pruned.onnx
        │
benchmark.py ─▶ reports/benchmark/ (JSON + PNG + MD)
```
Воспроизводимость: `dvc.yaml` фиксирует stages/deps/params/outs; метрики версионируются;
запуск в K8s — Job `model-retrain` (GPU-пул по nodeSelector в проде) из Airflow.

## 4. Петля мониторинга и переобучения (feedback loop)

```
 CronJob drift-check (ежечасно)
   ├─ Evidently: PSI по 20 признакам (ref=train, cur=свежие данные)
   ├─ AUC на current через ONNX → performance decay
   ├─ latest_drift.json ──▶ /app/reports PVC
   ├─ API /drift-status ──▶ Prometheus gauges ──▶ Алерты (drift>25%, AUC drop, stale)
   └─ exit 2 при алерте ──▶ Airflow DAG credit_scoring_retraining
                              ├─ check_data → check_drift (ветвление)
                              ├─ retrain (K8sJob, полный пайплайн)
                              ├─ ГЕЙТ: AUC(new) ≥ AUC(current) − 0.02
                              └─ canary-деплой (Istio 10→25→50→100% с анализом метрик)
```

## 5. Слои наблюдаемости

| Слой | Инструмент | Что смотрим |
|---|---|---|
| Метрики | Prometheus (ServiceMonitor) | RPS, p50/p95/p99, 5xx, инференс, HPA, cache hit, drift gauges |
| Логи | Loki + Promtail (JSON-парсинг) | структурированные логи API, уровни, ошибки |
| Дашборды | Grafana (2 каталоговых дашборда) | «API Performance», «Model Quality & Data Drift» |
| Алерты | PrometheusRule → Alertmanager → Slack/PD | 12 правил: доступность, SLO, модель, дрифт, ресурсы |
| Runbook | docs/runbook.md | диагностика и действия по каждому алерту |

## 6. Безопасность

- **Kubernetes**: non-root контейнеры, readOnlyRootFilesystem, drop ALL capabilities,
  seccomp RuntimeDefault, Pod Security (baseline enforce / restricted audit),
  NetworkPolicy default-deny + явные потоки (ingress→app, app→redis, prometheus scrape, DNS).
- **Облако (Terraform)**: KMS-шифрование etcd и бакетов, раздельные SA
  (cluster/nodes/CI с минимальными ролями), private node groups, API кластера в проде
  закрыт (`public_api=false`), admin CIDR-allowlist.
- **CI/CD**: gitleaks (секреты), bandit (SAST), pip-audit/trivy (уязвимости),
  checkov (IaC), manual approval + audit-лог для production.
- **Данные**: синтетический датасет; в проде — секреты только через K8s Secret/
  External Secrets, персональные данные не логируются.

## 7. Отказоустойчивость и масштабирование

| Компонент | Механизм |
|---|---|
| scoring-api | HPA 3→12 по CPU, topologySpread по узлам, PDB minAvailable=2 |
| Redis | опциональность кеша (degraded mode), LRU eviction, TTL |
| Кластер | cluster-autoscaler node group (Terraform, min/max по средам) |
| Контроллер | Liveness/readiness/startup-пробы, terminationGracePeriod 30 с |
| Релизы | rolling update (0 downtime), canary с автооткатом, rollback workflow |
| Модель | гейт паритета ONNX (CI), гейт AUC (Airflow), canary перед 100% трафика |

## 8. Матрица окружений

| | staging | production |
|---|---|---|
| Узлы CPU | 1–4, preemptible | 3–12, выделенные, 8 vCPU/16 ГБ |
| GPU-пул | нет | 1 × A100 (переобучение) |
| K8s API | публичный | закрытый (bastion/VPN) |
| Авто-обновления узлов | вкл | выкл (ручные окна) |
| Бюджетный алерт | 30k ₽ | 150k ₽ |
| Релиз | авто из develop/main | manual approval + canary |

## 9. Ответственность (владение)

| Компонент | Владелец | Эскалация |
|---|---|---|
| API инференс, модель | MLOps-команда | #oncall-ml |
| Инфраструктура (Terraform) | Platform-команда | #oncall-platform |
| CI/CD | MLOps-команда | #devops |
| Мониторинг/алерты | SRE + MLOps | #oncall-ml |
