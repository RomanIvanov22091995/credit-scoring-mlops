output "monitoring_sa_id" {
  value = yandex_iam_service_account.monitoring.id
}

output "monitoring_api_key" {
  value     = yandex_iam_service_account_api_key.monitoring.id
  sensitive = true
}

output "grafana_static_key_id" {
  value     = yandex_iam_service_account_static_access_key.grafana.access_key
  sensitive = true
}

output "grafana_static_secret" {
  value     = yandex_iam_service_account_static_access_key.grafana.secret_key
  sensitive = true
}
