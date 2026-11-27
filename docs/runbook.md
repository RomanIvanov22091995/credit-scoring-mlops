# Runbook: реагирование на инциденты Credit Scoring

Общие принципы:
1. Сначала диагностика через Grafana (`scoring-api-perf`) и Loki.
2. Действия — по симптомам ниже. Быстрые команды — в конце.
3. Все rollback-операции логируются (deploy-production.yml пишет audit-строку).

---

## ScoringApiDown — API недоступен

**Симптом:** алерт `ScoringApiDown` (critical), `up == 0` или отсутствие метрик.

**Диагностика:**
```bash
kubectl -n credit-scoring get pods -l app=scoring-api
kubectl -n credit-scoring describe pod -l app=scoring-api | tail -30
kubectl -n credit-scoring logs -l app=scoring-api --tail=100 --prefix
kubectl -n credit-scoring get events --sort-by=.lastTimestamp | tail -20
```

**Типовые причины и действия:**
| Причина | Признак | Действие |
|---|---|---|
| ImagePullBackOff | нет образа в registry | проверить тег в deployment; `kubectl rollout undo` |
| CrashLoopBackOff | исключение в логах | логи → fix → rollback `kubectl rollout undo deploy/scoring-api` |
| OOMKilled | `last state: terminated, reason: OOMKilled` | поднять limits.memory; проверить алерт `ScoringApiPodOOMKilled` |
| Все поды на одном узле и он умер | `kubectl get nodes` | node-pool autoscaler добавит узел; при необходимости `kubectl rollout restart deploy/scoring-api` |

---

## ScoringApiHigh5xxRate — доля ошибок > 2%

**Симптом:** алерт critical, `scoring_api_requests_total{status=~"5.."}` растёт.

**Шаги:**
1. Grafana → панель «Доля ошибок 5xx» + Loki: `{namespace="credit-scoring", pod=~"scoring-api.*"} | level="ERROR"`.
2. Если ошибки начались **после релиза** — немедленный rollback:
   ```bash
   kubectl -n credit-scoring rollout undo deployment/scoring-api
   ```
   или CI: Actions → rollback → `target: undo`.
3. Если ошибки с Redis (`redis connection error`) — проверить Redis:
   ```bash
   kubectl -n credit-scoring get pods -l app=redis
   kubectl -n credit-scoring logs -l app=redis --tail=50
   ```
   API работает без Redis (degraded mode) — Redis восстанавливать не блокируя скоринг.

---

## ScoringApiHighLatencyP95 — p95 > 50 мс

**Диагностика:**
1. Grafana «Реплики API (HPA)»: HPA упёрся в max → `ScoringApiHpaMaxedOut`?
2. CPU throttling: `kubectl top pods -n credit-scoring`.
3. Латентность именно инференса или HTTP-обвязки: панели «Латентность инференса» vs «Латентность HTTP».

**Действия:**
- Нагрузка: поднять `maxReplicas` HPA (`deployment/kubernetes/base/40-hpa.yaml`) или `kubectl scale`:
  ```bash
  kubectl -n credit-scoring patch hpa scoring-api-hpa \
    --type merge -p '{"spec":{"maxReplicas":20}}'
  ```
- Throttling: увеличить `limits.cpu` (0.5 → 1 CPU на под достаточно для ~800 rps).
- Регрессия новой модели: откатить модель (`deploy-production` → предыдущий тег).

---

## ScoringDataDriftDetected — дрифт данных

**Симптом:** `scoring_drift_share_drifted_features > 0.25` (критический алерт).

**Шаги:**
1. Открыть отчёт: `reports/drift/drift_report.html` (или Grafana → Model Quality & Data Drift).
2. Список дрейфовавших признаков и PSI: `reports/drift/latest_drift.json` → `drifted_features`, `drift_scores_psi`.
3. Понять источник: новое маркетинговое лендинг-предложение? Новой регион/сегмент? Поломанный ETL?
   - Поломка данных (ETL) → фикс ETL, пере-прогон: дрифт уйдёт сам.
   - Настоящий сдвиг рынка → запустить переобучение: Airflow DAG `credit_scoring_retraining`
     (`airflow dags trigger credit_scoring_retraining`) или CI-подобный K8s Job:
     ```bash
     kubectl -n credit-scoring create job --from=cronjob/drift-check drift-manual
     kubectl -n credit-scoring create job retrain-manual --image=credit-scoring-trainer:1.0.0
     ```
4. Новая модель проходит автоматическую валидацию (AUC-гейт) и выкатывается canary-цепочкой
   (`canary-release.yml`), НЕ сразу на 100% трафика.

---

## ScoringModelAucDrop — деградация качества

**Шаги:**
1. Сравнить `scoring_drift_test_auc` с максимумом в Grafana.
2. Проверить, не изменилась ли разметка/состав current-данных (ложный сигнал).
3. Подтверждённый провал → Airflow переобучение (см. выше) + canary-промоушен после A/B-отчёта.

---

## ScoringDriftCheckStale — проверка дрифта не выполняется

```bash
kubectl -n credit-scoring get cronjob drift-check
kubectl -n credit-scoring get jobs | grep drift-check
kubectl -n credit-scoring logs job/drift-check-XXXXX
```
Типовое: упал CronJob (нет reference/current CSV на PVC) — проверить, что Job переобучения
обновил `scoring-data-pvc`; вручную запустить job из cronjob.

---

## Экстренные команды

```bash
# Полный откат API
kubectl -n credit-scoring rollout undo deployment/scoring-api

# Откат к конкретной ревизии
kubectl -n credit-scoring rollout history deployment/scoring-api
kubectl -n credit-scoring rollout undo deployment/scoring-api --to-revision=5

# Рестарт (после починки конфигов)
kubectl -n credit-scoring rollout restart deployment/scoring-api

# Статус и события
kubectl -n credit-scoring get all
kubectl -n credit-scoring get events --sort-by=.lastTimestamp | tail -30
```

## Контакты
- Дежурный: канал `#oncall-ml`
- Владелец сервиса: команда MLOps
- Эскалация: см. `docs/architecture.md` (раздел «Ответственность»)
