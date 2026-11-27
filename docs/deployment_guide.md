# Deployment Guide — Credit Scoring System

## 0. Предварительные требования

| Инструмент | Версия | Для чего |
|---|---|---|
| Python | 3.11 | пайплайн модели, API |
| Docker | ≥24 | сборка образов, локальный kind |
| kubectl | ≥1.29 | работа с кластерами |
| kind | ≥0.25 | локальный тестовый кластер |
| helm | ≥3.14 | monitoring-стек |
| Terraform/OpenTofu | ≥1.5 | облако (Yandex Cloud) |
| yc CLI | — | bootstrap state, креды |

---

## 1. Локальный запуск модели и API (без кластера)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt -r requirements-api.txt -r requirements-dev.txt

python scripts/data_generation.py
python scripts/model_training/train_model.py          # ~30 с на CPU
python scripts/model_training/onnx_conversion.py
python scripts/model_training/onnx_validation.py      # гейт корректности
python scripts/model_training/optimize_model.py
python scripts/model_training/benchmark.py

uvicorn api.main:app --port 8000                       # Swagger: /docs
pytest tests/ -v
```

## 2. Сборка и проверка образов

```bash
docker build -f deployment/docker/Dockerfile.api      -t credit-scoring-api:1.0.0 .
docker build -f deployment/docker/Dockerfile.frontend -t credit-scoring-frontend:1.0.0 .
docker build -f deployment/docker/Dockerfile.trainer  -t credit-scoring-trainer:1.0.0 .
docker build -f deployment/docker/Dockerfile.drift    -t credit-scoring-drift:1.0.0 .

# smoke-тест API-образа
docker run -d --rm -p 8001:8000 credit-scoring-api:1.0.0
curl http://localhost:8001/health && curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' -d @tests/fixtures/payload.json
docker stop $(docker ps -q --filter ancestor=credit-scoring-api:1.0.0)
```

## 3. Локальный кластер kind (полный стек)

```bash
# 3.1 Кластер (см. kind-config.yaml: control-plane + worker, NodePort 30080→8080)
kind create cluster --name credit-scoring --config kind-config.yaml

# 3.2 Образы внутрь кластера
kind load docker-image credit-scoring-api:1.0.0 credit-scoring-frontend:1.0.0 \
  credit-scoring-drift:1.0.0 credit-scoring-trainer:1.0.0 --name credit-scoring

# 3.3 Ingress + мониторинг
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts && helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace \
  --set controller.service.type=NodePort --set controller.service.nodePorts.http=30080

helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f monitoring/helm/prometheus-stack-values.yaml

helm install loki grafana/loki-stack -n monitoring -f monitoring/helm/loki-values.yaml

# ВАЖНО: loki-stack создаёт datasource Loki с isDefault=true, что конфликтует
# с Prometheus (только один default на org). Правим и (опционально) ограничиваем
# шаблон в chart values:
kubectl -n monitoring patch cm loki-loki-stack --type merge -p \
  '{"data":{"loki-stack-datasource.yaml":"apiVersion: 1\ndatasources:\n- name: Loki\n  type: loki\n  access: proxy\n  url: http://loki:3100\n  version: 1\n  isDefault: false\n"}}'
kubectl -n monitoring delete pod -l app.kubernetes.io/name=grafana   # перечитать провижининг

# 3.4 Лейблы namespace для NetworkPolicy
kubectl label ns monitoring app.kubernetes.io/name=monitoring --overwrite
kubectl label ns ingress-nginx app.kubernetes.io/name=ingress-nginx --overwrite

# 3.5 Приложение
kubectl apply -k deployment/kubernetes/base
kubectl -n credit-scoring rollout status deployment/scoring-api

# 3.6 Дашборды Grafana (sidecar подхватит ConfigMap)
kubectl apply -f monitoring/grafana/dashboards-configmap.yaml
kubectl apply -f monitoring/alerts/scoring-alerts.yaml

# 3.7 Проверка
curl -H 'Host: scoring.example.com' http://localhost:8080/api/health
curl -H 'Host: scoring.example.com' http://localhost:8080/api/metrics | head -20
# UI Grafana: kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
#   (admin / пароль из секрета grafana-admin-credentials)

# 3.8 Подготовка PVC для drift-check и переобучения (данные/модель/отчёты)
python scripts/seed_pvcs.py
# kind local-path не поддерживает fsGroup — разово выдать права (в YC CSI не нужно):
kubectl -n credit-scoring run pvc-fix --image=busybox:1.36 -- sleep 60 -- \
  -m volumeMounts...   # см. scripts/seed_pvcs.py либо chmod 777 вручную
kubectl -n credit-scoring create job --from=cronjob/drift-check drift-check-manual
kubectl -n credit-scoring wait --for=condition=complete job/drift-check-manual --timeout=240s || \
  echo "exit 2 = алерт дрифта (ожидаемо на дрифтованных данных), отчёты в PVC"

# 3.9 Переобучение в кластере (полный пайплайн Stage 1 в K8s Job)
kubectl -n credit-scoring create job --from=job/model-retrain retrain-demo
kubectl -n credit-scoring logs job/retrain-demo -f | grep -E "TEST|ROC-AUC|ВСЕ ПРОВЕРКИ"
```

## 4. Облако Yandex Cloud (staging → production)

### 4.1 Backend для remote state (один раз на облако)

```bash
cd infrastructure/scripts
./bootstrap-state.sh <FOLDER_ID> staging
./bootstrap-state.sh <FOLDER_ID> production
```

### 4.2 Staging

```bash
export TF_VAR_yc_token="<oauth-token>"
export TF_VAR_ssh_public_key="$(cat ~/.ssh/id_ed25519.pub)"

cd infrastructure/environments/staging
terraform init            # remote state в Object Storage
terraform plan -out=tfplan
terraform apply tfplan    # VPC + K8s (KMS, SG) + бакеты + мониторинг SA

# Получить kubeconfig
yc kubernetes cluster get-credentials k8s-credit-scoring-staging --external
kubectl get nodes
```

### 4.3 Продакшен (отличия)

```bash
cd ../production
# public_api=false: доступ к API только из внутренних сетей/bastion
# GPU-пул включён (enable_gpu_pool=true) для переобучения
terraform init && terraform plan && terraform apply
yc kubernetes cluster get-credentials k8s-credit-scoring-production --internal
```

### 4.4 Registry и первый деплой

```bash
yc container registry create --name credit-scoring
REGISTRY_ID=$(yc container registry get credit-scoring --format json | jq -r .id)
docker login cr.yandex -u oauth -p "$TF_VAR_yc_token"

for img in api frontend trainer drift; do
  docker tag credit-scoring-$img:1.0.0 cr.yandex/$REGISTRY/credit-scoring-$img:1.0.0
  docker push cr.yandex/$REGISTRY/credit-scoring-$img:1.0.0
done

kubectl apply -k deployment/kubernetes/base
```

### 4.5 CI/CD

1. В Settings → Secrets: `KUBECONFIG_STAGING`, `KUBECONFIG_PRODUCTION`
   (`cat kubeconfig | base64`), `PROMETHEUS_URL`.
2. Branch protection на `main`: required reviews (production environment).
3. Push в `develop` → `build-and-test` → `deploy-staging` (автоматом).
4. Production: Actions → `deploy-production` → запустить с тегом и обоснованием.
5. Canary: Actions → `canary-release` → тег кандидата.

## 5. Airflow (переобучение)

```bash
# локальная проверка DAG
AIRFLOW_HOME=/tmp/airflow-home airflow db migrate
AIRFLOW_HOME=/tmp/airflow-home airflow dags list-import-errors   # должно быть пусто

# в кластере (официальный Helm chart)
helm repo add apache-airflow https://airflow.apache.org
helm install airflow apache-airflow/airflow \
  --set dags.gitSync.enabled=true \
  --set dags.gitSync.repo="https://github.com/<org>/credit-scoring-mlops" \
  --set dags.gitSync.subPath="airflow/dags" \
  -n airflow --create-namespace
airflow dags trigger credit_scoring_retraining   # ручной запуск; по дрифту — CronJob
```

## 6. Проверка (что считается «готово»)

- [x] `pytest tests/` — 10/10 passed
- [x] `onnx_validation.py` exit 0 (паритет 1e-7, AUC match)
- [x] Terraform validate — Success (staging, production, 4 модуля)
- [x] 4 образа собираются; API-образ отвечает на /health, /predict
- [x] kind-стек: ingress → api → redis, HPA/CronJob/ServiceMonitor созданы
- [x] Prometheus скрейпит /metrics, дашборды Grafana видны, алерты загружены
- [x] Evidently: drifted → alert (exit 2), stable → 0% дрифта
- [x] Airflow DAG импортируется без ошибок (10 задач)
