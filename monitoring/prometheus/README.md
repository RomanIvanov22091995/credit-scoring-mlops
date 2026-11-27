# Prometheus — где что лежит

Prometheus в этом проекте **не конфигурируется файлами из этой папки** — он
устанавливается Helm-чартом `kube-prometheus-stack`, и вся настройка идёт через values:

| Что | Где |
|---|---|
| Установка и настройка (retention, storage, remote write, Alertmanager) | `monitoring/helm/prometheus-stack-values.yaml` |
| Правила алертов (PrometheusRule) | `monitoring/alerts/scoring-alerts.yaml` |
| Автоскрейпинг приложения | `deployment/kubernetes/base/80-servicemonitor.yaml` (CRD ServiceMonitor) |
| Метрики приложения (экспортёр) | `api/monitoring.py` + эндпоинт `api/main.py: /metrics` |

## Как Prometheus находит наши метрики (цепочка)

```
api/monitoring.py  ──▶  /metrics (текстовый формат exposition)
deployment 20-...yaml  ──▶  аннотации prometheus.io/scrape
80-servicemonitor.yaml  ──▶  CRD ServiceMonitor (label selector)
Prometheus Operator  ──▶  генерирует конфиг Prometheus
Prometheus  ──▶  pull GET /metrics каждые 15s
```

Ключевая идея: **pull-модель** — сервер сам опрашивает приложения, а не наоборот.
Правило `ruleSelectorNilUsesHelmValues: false` в values разрешает оператору подбирать
PrometheusRule из любых namespace.

## Автономный Prometheus без кластера (отладка)

`prometheus-standalone.yml` в этой папке — конфиг для локального запуска в docker
(скрейпит API на localhost:8000):

```bash
docker run -d --name prom -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus/prometheus-standalone.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
# UI: http://localhost:9090 ; проверить таргет Status → Targets
```
