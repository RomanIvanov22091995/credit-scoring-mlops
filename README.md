# Credit Scoring MLOps — промышленное развертывание кредитной скоринговой системы

Полный MLOps-цикл для кредитного скоринга: **NN-модель → ONNX → контейнер → Kubernetes → CI/CD → мониторинг → дрифт → авто-переобучение**.

Проект не только написан, но и **исполнен**: полный стек деплоен в локальный kind-кластер,
все пайплайны отработали (см. колонку «Проверено»).

| Этап проекта | Статус | Проверено запуском |
|---|---|---|
| 1. Модель (ONNX, INT8, pruning, benchmark) | ✅ | паритет 1.5e-07, INT8 ×3.24, НТ 866 rps |
| 2. Terraform (Yandex Cloud, модули, remote state) | ✅ | `validate: VALID` — 4 модуля + 2 окружения |
| 3. Docker + Kubernetes | ✅ | 4 образа; стек работает в kind (API/Redis/ingress/HPA) |
| 4. CI/CD (build→test→scan→deploy→monitor) | ✅ | 5 workflow, линтеры YAML, гейты в коде |
| 5. Мониторинг | ✅ | Prometheus скрейпит API; 12 алертов загружены; 2 дашборда в Grafana; Loki собирает логи |
| 6. Дрифт + Seldon + Istio | ✅ | drift-check Job в K8s: отчёты в PVC, гейдж 0.381 в Prometheus, алерт |
| 7. Автоматизация | ✅ | DAG в Airflow 2.10.5 (0 ошибок импорта); retrain Job в K8s: полный пайплайн, AUC 0.7365 |

---

## 1. Модель и её производительность

- **Архитектура**: PyTorch MLP `20 → 256 → 128 → 64 → 1` (BatchNorm + ReLU + Dropout),
  задача — бинарная классификация дефолта по 20 признакам заёмщика.
- **Качество**: test ROC-AUC **0.736**, KS **0.35**, PR-AUC **0.40** (`models/trained/metrics.json`).
- **Конвертация**: `torch.onnx.export` (opset 18, динамический batch) — ускорение инференса
  **×8.2** против PyTorch на CPU, максимальное расхождение вероятностей **1.5e-07**
  (`models/production/validation_report.json`).
- **Оптимизация**: INT8-динамическая квантизация — **сжатие ×3.24** (195 → 60 КБ) при
  ΔAUC = 8e-06; L1-pruning 30% — без потери AUC (`optimization_report.json`).

### Быстрый старт (локально)

```bash
# 1. Окружение
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements-ml.txt -r requirements-api.txt -r requirements-dev.txt

# 2. Пайплайн модели (данные → обучение → ONNX → валидация → оптимизация → benchmark)
python scripts/data_generation.py
python scripts/model_training/train_model.py
python scripts/model_training/onnx_conversion.py
python scripts/model_training/onnx_validation.py     # CI-гейт: exit != 0 при расхождении
python scripts/model_training/optimize_model.py      # INT8 + pruning
python scripts/model_training/benchmark.py           # матрица batch × threads × варианты

# 3. API локально
uvicorn api.main:app --port 8000
# Swagger: http://localhost:8000/docs ; метрики: /metrics ; дрифт: /drift-status

# 4. Тесты
pytest tests/ -v

# 5. Дрифт-мониторинг (изолированное окружение: pandas<3)
uv pip install --python .venv-drift -r requirements-drift.txt  # или pip в venv
.venv-drift/bin/python scripts/drift/drift_monitoring.py       # дрифтованные данные → exit 2
.venv-drift/bin/python scripts/drift/drift_monitoring.py --current data/reference/current_stable.csv

# 6. Нагрузочный тест
locust -f scripts/model_training/load_test.py --host=http://localhost:8000 --headless -u 50 -r 10 -t 60s
```

### Результаты benchmark (Apple M-series, CPUExecutionProvider, 1 поток)

| Вариант | Размер | b=1 p50 | AUC (test) |
|---|---|---|---|
| pytorch_fp32 | 192 КБ | 0.036 мс | 0.73576 |
| **onnx_fp32** | 195 КБ | **0.005 мс** | 0.73576 |
| **onnx_int8** | **60 КБ** | 0.007 мс | 0.73577 |
| onnx_pruned_30 | 195 КБ | 0.005 мс | 0.73584 |

Нагрузочный тест API (50 виртуальных пользователей, 45 с): **866 rps**, 0 ошибок,
p50 = 1 мс, **p95 = 3 мс**, p99 = 13 мс — SLO 50 мс выполнен с запасом
(`reports/load_tests/`).

---

## 2. Инфраструктура (Terraform, Yandex Cloud)

```
infrastructure/
├── modules/            # network (VPC/SG/NAT), kubernetes (MK8s+KMS+CPU/GPU pools),
│                       # storage (Object Storage: state/DVC/MLflow), monitoring (SA/API-ключи)
├── environments/       # staging и production (remote state в Object Storage)
└── scripts/            # bootstrap-state.sh — создание backend-бакета и ключей
```

Ключевые свойства: модульность, KMS-шифрование etcd и бакетов, раздельные сервисные
аккаунты кластера/узлов/CI, security groups с запретом внешнего входа на узлы,
preemptible-узлы только в staging, автоскейлинг CPU-пулов и GPU-пул для обучения.

```bash
cd infrastructure/environments/staging
./../../infrastructure/scripts/bootstrap-state.sh <folder_id> staging   # один раз
export TF_VAR_yc_token=... TF_VAR_ssh_public_key=...
terraform init && terraform plan && terraform apply
```

**Проверка без облака**: `terraform validate` — Success для staging, production и всех
4 модулей (выполнено; см. docs/deployment_guide.md, раздел «Проверка»).

---

## 3. Контейнеры и Kubernetes

| Образ | Стадии | Итоговый размер |
|---|---|---|
| `credit-scoring-api` | builder → runtime (non-root, healthcheck, read-only fs) | 178 МБ |
| `credit-scoring-frontend` | nginx-unprivileged, статика + proxy | 22 МБ |
| `credit-scoring-trainer` | builder → **dvc** (pull данных из S3) → final | ~1.6 ГБ |
| `credit-scoring-drift` | Evidently + pandas<3 (изоляция от API) | 316 МБ |

Kubernetes-манифесты (`deployment/kubernetes/base/`, применяются kustomize):
Deployment **rolling update** (maxSurge=1/maxUnavailable=0), три пробы
(startup/readiness/liveness), ConfigMap + Secret, Service + Ingress, **HPA** (CPU 65%,
3→12 реплик, аккуратный scale-down), NetworkPolicy default-deny + явные разрешения,
PDB, Redis-кеш, CronJob дрифта, Job переобучения, ServiceMonitor.

**Локальный деплой в kind** (весь стек проверен):

```bash
kind create cluster --config kind-config.yaml
kind load docker-image credit-scoring-api:1.0.0 credit-scoring-frontend:1.0.0 --name credit-scoring
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace \
  --set controller.service.type=NodePort --set controller.service.nodePorts.http=30080
kubectl apply -k deployment/kubernetes/base
curl -H 'Host: scoring.example.com' http://localhost:8080/api/health
```

---

## 4. CI/CD (GitHub Actions)

```
.github/workflows/
├── build-and-test.yml      # test (данные→модель→ONNX→pytest) + security (bandit, pip-audit,
│                           # gitleaks, checkov, trivy) + IaC (terraform fmt/validate, kubeconform)
│                           # + docker buildx (4 образа, кеш GHA) + Trivy-гейт
├── deploy-staging.yml      # автодеплой из develop/main: kustomize → rollout → smoke →
│                           # нагрузочный гейт (p95<50ms) → авто-rollback при провале
├── deploy-production.yml   # manual approval (environment) → verify → canary-наблюдение
├── canary-release.yml      # Istio: 10% → 25% → 50% → 100% c анализом метрик на каждом шаге
└── rollback.yml            # undo/revision/image + scheduled health-watch каждые 15 мин
```

Секреты: `KUBECONFIG_STAGING`, `KUBECONFIG_PRODUCTION` (base64 kubeconfig), `PROMETHEUS_URL`.

---

## 5. Мониторинг и observability

- **kube-prometheus-stack** (Prometheus + Grafana + Alertmanager) и **loki-stack**
  (Loki + Promtail с парсингом JSON-логов) — values в `monitoring/helm/`.
- **Метрики приложения** (`api/monitoring.py`): RPS по эндпоинтам, латентность HTTP/инференса
  (histogram p50/p95/p99), счётчик предсказаний, распределение вероятности дефолта,
  Redis hit ratio, версия модели (gauge `scoring_model_info`), дрифт-гейджи.
- **Дашборды Grafana** (`monitoring/grafana/dashboards/`, sidecar-подгрузка):
  «API Performance» (10 панелей) и «Model Quality & Data Drift» (6 панелей, включая Loki-логи).
- **Алерты** (`monitoring/alerts/scoring-alerts.yaml`): API down, 5xx>2%, p95>50мс,
  HPA упёрся, OOMKilled, stall предсказаний, **data drift >25%**, **AUC drop >0.05**,
  устаревший drift-check. Runbook — `docs/runbook.md`.

---

## 6. Мониторинг дрифта и управление моделями

- **Evidently** (`scripts/drift/drift_monitoring.py`): PSI-статтестат по всем 20 признакам,
  качество на current через ONNX (AUC), HTML-отчёты (`reports/drift/*.html`) и
  машинночитаемый `latest_drift.json` (его читают API `/drift-status`, Prometheus-гейджи и Airflow).
  Проверено: дрифтованная выборка → **38.1% признаков дрейфуют, alert (exit 2)**;
  стабильная → **0%**.
- **Seldon Core** (`seldon/`): A/B-тест 75/25 (fp32 vs int8), **shadow deployment** (кандидат
  обрабатывает трафик-тень, клиенту отдаётся только main).
- **Istio canary** (`istio/canary-rollout.yaml`): DestinationRule subsets + VirtualService
  90/10 с mTLS и повторами; веса управляются `canary-release.yml`.

---

## 7. Пайплайн переобучения и автоматизация

- **Airflow DAG** `credit_scoring_retraining` (`airflow/dags/retraining_dag.py`):
  проверка свежести данных → ветвление по дрифту (порог 25%) → K8sJob переобучения
  (полный пайплайн Stage 1) → **гейт качества** (AUC новой >= AUC текущей − 0.02) →
  canary-деплой → notify. Валидирован: `airflow dags list-import-errors` — пусто,
  10 задач (`airflow tasks list`). Триггер по дрифту: `trigger_retraining_if_drifted()`.
- **Переобучение в K8s проверено**: Job `model-retrain` исполнил полный пайплайн в кластере
  (данные → обучение → ONNX → валидация → INT8/pruning): test ROC-AUC **0.73651**,
  сжатие INT8 **×3.25**, pruning 30% разреженности, все гейты пройдены.
- **Замкнутая петля дрифта** (проверено в kind): CronJob drift-check → Evidently-отчёт в PVC
  → API `/drift-status` = `drift_detected` (share 0.381, AUC 0.7509) → гейдж
  `scoring_drift_share_drifted_features` в Prometheus → алерт `ScoringDataDriftDetected`
  → триггер Airflow-переобучения.
- **Масштабирование**: HPA (CPU/memory) + cluster-autoscaler node groups
  (Terraform: min 1–3, max 4–12) + topologySpreadConstraints по узлам.
- **Кеширование**: Redis (allkeys-lru, TTL 300 с) — проверено в кластере: повторный запрос
  `cached: true`, латентность 0.28 мс; при недоступности Redis API работает без кеша.

---

## Документация

- [docs/architecture.md](docs/architecture.md) — архитектура системы, потоки данных, отказоустойчивость.
- [docs/deployment_guide.md](docs/deployment_guide.md) — пошаговый деплой (стейджинг/прод/локально).
- [docs/runbook.md](docs/runbook.md) — реагирование на инциденты.
- [docs/benchmark_report.md](docs/benchmark_report.md) — полный отчёт о производительности.

## Структура репозитория

```
├── api/                    # FastAPI-сервис инференса (ONNX, метрики, Redis-кеш)
├── frontend/               # nginx-frontend (форма заявки → решение)
├── scripts/                # data_generation, model_training/*, drift/*
├── models/                 # trained (pt/scaler) + production (onnx, отчёты)
├── tests/                  # pytest: API-контракты, паритет ORT, качество, DAG
├── config/                 # params.yaml, feature_names.json
├── deployment/
│   ├── docker/             # Dockerfile.api / .frontend / .trainer / .drift
│   └── kubernetes/base/    # 13 манифестов + kustomization
├── infrastructure/         # Terraform: modules/ + environments/ + scripts/
├── monitoring/             # helm values, алерты, дашборды Grafana, Loki
├── airflow/dags/           # DAG переобучения
├── seldon/, istio/         # A/B, shadow, canary
├── .github/workflows/      # 5 CI/CD workflow
├── reports/                # benchmark, drift, load_tests (артефакты прогонов)
└── docs/                   # architecture, deployment guide, runbook, benchmark
```

## Требования

- Локально: Python 3.11, Docker, kubectl, kind, helm, locust (для НТ).
- Облако: Yandex Cloud (yc CLI), Terraform ≥1.5, домен + cert-manager для TLS.
