# =============================================================================
# Модуль monitoring: аккаунты и права для мониторинга.
# Алертинг уровня облака (бюджеты) настраивается в Yandex Monitoring через
# CLI/UI (нет ресурса в open-source провайдере) — см. docs/deployment_guide.md.
# Основной стек наблюдаемости (Prometheus/Grafana/Loki) разворачивается Helm'ом
# в кластере — см. monitoring/.
# =============================================================================
terraform {
  required_version = ">= 1.5"
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.130"
    }
  }
}

# Сервисный аккаунт мониторинга (remote-write из Prometheus в Yandex Monitoring)
resource "yandex_iam_service_account" "monitoring" {
  name        = "sa-${var.project}-${var.environment}-monitoring"
  description = "Запись метрик кластера в Yandex Monitoring"
}

resource "yandex_iam_service_account_api_key" "monitoring" {
  service_account_id = yandex_iam_service_account.monitoring.id
  description        = "API-ключ для Prometheus remote write"
}

resource "yandex_resourcemanager_folder_iam_member" "monitoring" {
  folder_id = var.folder_id
  role      = "monitoring.editor"
  member    = "serviceAccount:${yandex_iam_service_account.monitoring.id}"
}

# Аккаунт Grafana: только чтение метрик облака
resource "yandex_iam_service_account" "grafana" {
  name        = "sa-${var.project}-${var.environment}-grafana"
  description = "Чтение метрик Yandex Monitoring для дашбордов Grafana"
}

resource "yandex_iam_service_account_static_access_key" "grafana" {
  service_account_id = yandex_iam_service_account.grafana.id
  description        = "Static key Grafana -> Yandex Monitoring"
}

resource "yandex_resourcemanager_folder_iam_member" "grafana" {
  folder_id = var.folder_id
  role      = "monitoring.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.grafana.id}"
}
