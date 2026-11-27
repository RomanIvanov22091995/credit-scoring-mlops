# =============================================================================
# Модуль kubernetes: Managed K8s, CPU/GPU node groups, KMS-шифрование
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

# Сервисные аккаунты: один для кластера, отдельный для узлов (минимальные права)
resource "yandex_iam_service_account" "cluster" {
  name        = "sa-${var.project}-${var.environment}-k8s"
  description = "Сервисный аккаунт Managed Kubernetes ${var.environment}"
}

resource "yandex_iam_service_account" "nodes" {
  name        = "sa-${var.project}-${var.environment}-nodes"
  description = "Сервисный аккаунт worker-узлов ${var.environment}"
}

resource "yandex_resourcemanager_folder_iam_member" "cluster" {
  for_each = toset([
    "k8s.clusters.agent",
    "vpc.publicAdmin",
    "load-balancer.admin",
  ])
  folder_id   = var.folder_id
  role        = each.value
  member      = "serviceAccount:${yandex_iam_service_account.cluster.id}"
  sleep_after = 30
}

resource "yandex_resourcemanager_folder_iam_member" "nodes" {
  for_each = toset([
    "k8s.clusters.agent",
    "container-registry.editor",
    "yc.containerRegistry.infra",
    "storage.viewer",
  ])
  folder_id   = var.folder_id
  role        = each.value
  member      = "serviceAccount:${yandex_iam_service_account.nodes.id}"
  sleep_after = 30
}

# KMS-ключ для шифрования секретов etcd и загрузочных дисков
resource "yandex_kms_symmetric_key" "kms" {
  name              = "kms-${var.project}-${var.environment}"
  description       = "Шифрование секретов кластера ${var.environment}"
  default_algorithm = "AES_128"
  rotation_period   = "8760h" # годовая ротация
}

resource "yandex_kms_symmetric_key_iam_binding" "kms_encrypter" {
  symmetric_key_id = yandex_kms_symmetric_key.kms.id
  role             = "kms.keys.encrypterForSubject"
  members          = ["serviceAccount:${yandex_iam_service_account.nodes.id}"]
}

# ---------------------------- Managed Kubernetes ----------------------------
resource "yandex_kubernetes_cluster" "this" {
  name        = "k8s-${var.project}-${var.environment}"
  description = "Managed Kubernetes для ${var.project} (${var.environment})"
  network_id  = var.network_id

  labels = var.labels

  master {
    version   = var.k8s_version
    public_ip = var.public_api

    master_location {
      zone      = var.master_zone
      subnet_id = var.master_subnet_id
    }

    maintenance_policy {
      auto_upgrade = var.environment != "production"
      maintenance_window {
        start_time = "03:00"
        duration   = "3h"
      }
    }

    security_group_ids = [var.k8s_security_group_id]
  }

  service_account_id      = yandex_iam_service_account.cluster.id
  node_service_account_id = yandex_iam_service_account.nodes.id

  kms_provider {
    key_id = yandex_kms_symmetric_key.kms.id
  }
}

# ---------------------------- CPU node group (автоскейлинг) ----------------------------
resource "yandex_kubernetes_node_group" "cpu" {
  cluster_id  = yandex_kubernetes_cluster.this.id
  name        = "cpu-${var.project}-${var.environment}"
  description = "CPU-узлы: инференс API, мониторинг, вспомогательные сервисы"

  labels = merge(var.labels, { node-pool = "cpu" })

  instance_template {
    platform_id = var.cpu_platform_id

    resources {
      memory = var.cpu_memory
      cores  = var.cpu_cores
    }

    boot_disk {
      type = "network-ssd"
      size = var.cpu_disk_size
    }

    scheduling_policy {
      # production = не вытесняемые; staging = preemptible (экономия)
      preemptible = var.environment != "production"
    }

    network_interface {
      subnet_ids         = var.node_subnet_ids
      nat                = false
      security_group_ids = [var.nodes_security_group_id]
    }

    container_runtime {
      type = "containerd"
    }

    metadata = {
      ssh-keys = "ubuntu:${var.ssh_public_key}"
    }
  }

  scale_policy {
    auto_scale {
      min     = var.cpu_nodes_min
      max     = var.cpu_nodes_max
      initial = var.cpu_nodes_initial
    }
  }

  allocation_policy {
    location {
      zone = var.master_zone
    }
  }

  maintenance_policy {
    auto_upgrade = var.environment != "production"
    auto_repair  = true
    maintenance_window {
      start_time = "03:00"
      duration   = "3h"
    }
  }
}

# ---------------------------- GPU node group ----------------------------
resource "yandex_kubernetes_node_group" "gpu" {
  count = var.enable_gpu_pool ? 1 : 0

  cluster_id  = yandex_kubernetes_cluster.this.id
  name        = "gpu-${var.project}-${var.environment}"
  description = "GPU-узлы: переобучение модели, тяжёлый инференс"

  labels = merge(var.labels, { node-pool = "gpu" })

  instance_template {
    platform_id = var.gpu_platform_id

    resources {
      memory = var.gpu_memory
      cores  = var.gpu_cores
      gpus   = var.gpu_count # A100: по спецификации платформы gpu-standard-v3
    }

    boot_disk {
      type = "network-ssd"
      size = var.gpu_disk_size
    }

    scheduling_policy {
      preemptible = false # GPU-узлы в проде не вытесняем
    }

    network_interface {
      subnet_ids         = var.node_subnet_ids
      nat                = false
      security_group_ids = [var.nodes_security_group_id]
    }

    container_runtime {
      type = "containerd"
    }

    metadata = {
      ssh-keys = "ubuntu:${var.ssh_public_key}"
    }
  }

  scale_policy {
    fixed_scale {
      size = var.gpu_nodes
    }
  }

  allocation_policy {
    location {
      zone = var.gpu_zone
    }
  }

  maintenance_policy {
    auto_upgrade = var.environment != "production"
    auto_repair  = true
  }
}
