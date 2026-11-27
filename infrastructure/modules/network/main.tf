# =============================================================================
# Модуль network: VPC, подсети, security groups, NAT для Yandex Cloud
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

resource "yandex_vpc_network" "this" {
  name        = "vpc-${var.project}-${var.environment}"
  description = "Сеть для проекта ${var.project} (${var.environment})"
  labels      = var.labels
}

# Основные подсети в 3 зонах доступности
resource "yandex_vpc_subnet" "public" {
  for_each = { for idx, cfg in var.zones : cfg.zone => cfg }

  name           = "subnet-${var.project}-${var.environment}-${each.key}"
  zone           = each.key
  network_id     = yandex_vpc_network.this.id
  v4_cidr_blocks = [each.value.cidr]
  labels         = var.labels
}

# NAT-шлюз для исходящего доступа приватных узлов
resource "yandex_vpc_gateway" "nat" {
  name = "gw-${var.project}-${var.environment}-nat"
  shared_egress_gateway {}
}

resource "yandex_vpc_route_table" "nat" {
  name       = "rt-${var.project}-${var.environment}-nat"
  network_id = yandex_vpc_network.this.id

  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat.id
  }
}

# Security group: вход в кластер и сервисы только с нужных диапазонов
resource "yandex_vpc_security_group" "k8s" {
  name        = "sg-${var.project}-${var.environment}-k8s"
  network_id  = yandex_vpc_network.this.id
  description = "Security group Managed K8s"

  labels = var.labels

  # Внутрикластерный трафик разрешён полностью
  ingress {
    protocol          = "ANY"
    description       = "Внутрикластерный трафик"
    predefined_target = "security_group_id"
    security_group_id = yandex_vpc_security_group.k8s.id
  }

  # Kubernetes API (6443) — из админских подсетей
  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    port           = 6443
    v4_cidr_blocks = var.admin_cidrs
  }

  # NodePort / Ingress — из интернета с ограничением диапазонов LB
  ingress {
    protocol       = "TCP"
    description    = "HTTP ingress"
    port           = 80
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTPS ingress"
    port           = 443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  # Health checks Yandex LB
  ingress {
    protocol       = "TCP"
    description    = "Health checks NLB"
    port           = 30500
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Исходящий трафик (нат, обновления, реестры)"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

# Отдельная группа для worker-узлов (закрыт прямой вход извне)
resource "yandex_vpc_security_group" "nodes" {
  name        = "sg-${var.project}-${var.environment}-nodes"
  network_id  = yandex_vpc_network.this.id
  description = "Security group worker-узлов"

  labels = var.labels

  ingress {
    protocol          = "ANY"
    description       = "Трафик от security group кластера"
    security_group_id = yandex_vpc_security_group.k8s.id
  }

  ingress {
    protocol          = "ANY"
    description       = "Внутригрупповой трафик узлов"
    predefined_target = "security_group_id"
    security_group_id = yandex_vpc_security_group.nodes.id
  }

  ingress {
    protocol       = "TCP"
    description    = "NodePort для health check NLB"
    port           = 30500
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "Исходящий трафик через NAT"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}
