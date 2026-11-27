# Loki — где что лежит

Loki (централизованное хранилище логов) тоже устанавливается Helm-чартом:

| Что | Где |
|---|---|
| Установка (retention, PVC, ресурсы) | `monitoring/helm/loki-values.yaml` |
| Сбор логов из подов | внутри чарта: **Promtail** (DaemonSet) → `http://loki:3100/loki/api/v1/push` |
| Парсинг JSON-логов API в поля (level и т.д.) | `monitoring/helm/loki-values.yaml → promtail.config.snippets.pipelineStages` |
| Датасорс в Grafana | ConfigMap `loki-loki-stack` (поставляется чартом; `isDefault: false`!) |
| Лог-панель дашборда | `monitoring/grafana/dashboards/model-drift.json` (панель «Логи API») |

## Ключевое отличие Loki от ELK

Loki **не индексирует текст логов** — индексируются только **лейблы**
(namespace, pod, level...). Поиск по содержимому — фильтрами на этапе запроса
(LogQL). Это дёшево по хранению и быстро по лейблам; ELK (Elasticsearch)
индексирует весь текст — мощнее поиск, дороже хранение.

## LogQL — минимальный набор

```logql
{namespace="credit-scoring", pod=~"scoring-api.*"}        # все логи API
{namespace="credit-scoring"} | level="ERROR"               # после парсинга JSON
{namespace="credit-scoring"} |= "Redis"                    # substring-фильтр
rate({namespace="credit-scoring"} | level="ERROR" [5m])    # частота ошибок
```

## Проверка, что логи текут (запущено в kind, проверено)

```bash
kubectl -n monitoring port-forward svc/loki 3100:3100
curl -G -s --data-urlencode 'query={namespace="credit-scoring"}' \
  --data-urlencode 'limit=5' http://localhost:3100/loki/api/v1/query_range
```

## Автономный Loki без кластера (отладка)

`loki-local-config.yaml` — конфиг single-binary для локального docker:

```bash
docker run -d --name loki -p 3100:3100 \
  -v $(pwd)/monitoring/loki/loki-local-config.yaml:/etc/loki/local-config.yaml \
  grafana/loki:3.4.2 -config.file=/etc/loki/local-config.yaml
```
