output "tf_state_bucket" {
  value = yandex_storage_bucket.tf_state.bucket
}

output "dvc_bucket" {
  value = yandex_storage_bucket.dvc.bucket
}

output "mlflow_bucket" {
  value = yandex_storage_bucket.mlflow.bucket
}

output "ci_access_key_id" {
  value     = yandex_iam_service_account_static_access_key.ci.access_key
  sensitive = true
}

output "ci_secret_key" {
  value     = yandex_iam_service_account_static_access_key.ci.secret_key
  sensitive = true
}
