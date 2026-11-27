# =============================================================================
# Production: высокая доступность, закрытый API, GPU-пул для переобучения
# =============================================================================
terraform {
  required_version = ">= 1.5"

  backend "s3" {
    bucket                      = "credit-scoring-tfstate-production"
    key                         = "production/terraform.tfstate"
    region                      = "ru-central1"
    endpoint                    = "https://storage.yandexcloud.net"
    skip_region_validation      = true
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    force_path_style            = true
  }

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.130"
    }
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.master_zone
}

locals {
  labels = {
    project     = "credit-scoring"
    environment = var.environment
    managed_by  = "terraform"
    repo        = "credit-scoring-mlops"
    criticality = "high"
  }
}

module "network" {
  source = "../../modules/network"

  project     = var.project
  environment = var.environment
  labels      = local.labels
  admin_cidrs = var.admin_cidrs
}

module "kubernetes" {
  source = "../../modules/kubernetes"

  project     = var.project
  environment = var.environment
  folder_id   = var.folder_id
  labels      = local.labels

  network_id              = module.network.network_id
  master_subnet_id        = module.network.public_subnet_ids["a"]
  node_subnet_ids         = values(module.network.public_subnet_ids)
  k8s_security_group_id   = module.network.k8s_security_group_id
  nodes_security_group_id = module.network.nodes_security_group_id
  master_zone             = var.master_zone
  public_api              = false # доступ к API только через внутреннюю сеть/bastion
  registry_id             = var.registry_id
  ssh_public_key          = var.ssh_public_key

  # Production: автоскейлинг 3..12, preemptible выключен (в модуле), GPU-пул
  cpu_nodes_min     = 3
  cpu_nodes_max     = 12
  cpu_nodes_initial = 3
  cpu_cores         = 8
  cpu_memory        = 16
  enable_gpu_pool   = true
  gpu_nodes         = 1
}

module "storage" {
  source = "../../modules/storage"

  project     = var.project
  environment = var.environment
  folder_id   = var.folder_id
  kms_key_id  = module.kubernetes.kms_key_id
  labels      = local.labels
}

module "monitoring" {
  source = "../../modules/monitoring"

  project     = var.project
  environment = var.environment
  folder_id   = var.folder_id
}
