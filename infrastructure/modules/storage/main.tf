# =============================================================================
# Модуль storage: Object Storage — remote state Terraform, DVC remote,
# артефакты MLflow/моделей. Публичный доступ запрещён, шифрование через KMS.
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

# Бакет для remote state Terraform
resource "yandex_storage_bucket" "tf_state" {
  bucket   = "${var.project}-tfstate-${var.environment}"
  max_size = var.state_bucket_max_size_gb * 1024 * 1024 * 1024 # лимит защиты от утечки
  versioning {
    enabled = true # история стейтов для отката
  }
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = var.kms_key_id != "" ? var.kms_key_id : null
        sse_algorithm     = var.kms_key_id != "" ? "aws:kms" : "AES256"
      }
    }
  }
  force_destroy = var.environment == "staging"
}

# Бакет для DVC (данные и модели) с жизненным циклом
resource "yandex_storage_bucket" "dvc" {
  bucket = "${var.project}-dvc-${var.environment}"

  versioning {
    enabled = true
  }
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = var.kms_key_id != "" ? var.kms_key_id : null
        sse_algorithm     = var.kms_key_id != "" ? "aws:kms" : "AES256"
      }
    }
  }
  force_destroy = var.environment == "staging"

  dynamic "lifecycle_rule" {
    for_each = var.dvc_expiration_days > 0 ? [1] : []
    content {
      enabled = true
      expiration {
        days = var.dvc_expiration_days
      }
      noncurrent_version_expiration {
        days = max(var.dvc_expiration_days / 2, 7)
      }
    }
  }
}

# Бакет артефактов MLflow (модели, метрики)
resource "yandex_storage_bucket" "mlflow" {
  bucket = "${var.project}-mlflow-${var.environment}"

  force_destroy = var.environment == "staging"
}

# Сервисный аккаунт для CI (GitHub Actions) с доступом к бакетам
resource "yandex_iam_service_account" "ci" {
  name        = "sa-${var.project}-${var.environment}-ci"
  description = "CI: доступ к Object Storage (state, DVC)"
}

resource "yandex_iam_service_account_static_access_key" "ci" {
  service_account_id = yandex_iam_service_account.ci.id
  description        = "Static key для CI-доступа к Object Storage"
}

resource "yandex_resourcemanager_folder_iam_member" "ci" {
  for_each = toset([
    "storage.editor",
    "kms.keys.encrypterDecrypter",
  ])
  folder_id = var.folder_id
  role      = each.value
  member    = "serviceAccount:${yandex_iam_service_account.ci.id}"
}
