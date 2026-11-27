variable "project" {
  type        = string
  description = "Имя проекта"
}

variable "environment" {
  type        = string
  description = "staging | production"
}

variable "folder_id" {
  type        = string
  description = "ID каталога Yandex Cloud"
}

variable "network_id" {
  type        = string
  description = "ID VPC (модуль network)"
}

variable "master_subnet_id" {
  type        = string
  description = "Подсеть мастера"
}

variable "node_subnet_ids" {
  type        = list(string)
  description = "Подсети worker-узлов"
}

variable "k8s_security_group_id" {
  type        = string
  description = "SG кластера (модуль network)"
}

variable "nodes_security_group_id" {
  type        = string
  description = "SG узлов (модуль network)"
}

variable "k8s_version" {
  type        = string
  description = "Версия Kubernetes"
  default     = "1.29"
}

variable "master_zone" {
  type    = string
  default = "ru-central1-a"
}

variable "public_api" {
  type        = bool
  description = "Публичный IP API кластера (для production=false доступ через bastion)"
  default     = true
}

variable "labels" {
  type    = map(string)
  default = {}
}

# --- CPU pool ---
variable "cpu_platform_id" {
  type    = string
  default = "standard-v3"
}

variable "cpu_cores" {
  type    = number
  default = 4
}

variable "cpu_memory" {
  type    = number
  default = 8
}

variable "cpu_disk_size" {
  type    = number
  default = 64
}

variable "cpu_nodes_min" {
  type    = number
  default = 2
}

variable "cpu_nodes_max" {
  type    = number
  default = 10
}

variable "cpu_nodes_initial" {
  type    = number
  default = 2
}

# --- GPU pool ---
variable "enable_gpu_pool" {
  type        = bool
  description = "Создавать ли GPU node group (для переобучения)"
  default     = false
}

variable "gpu_platform_id" {
  type    = string
  default = "gpu-standard-v3"
}

variable "gpu_cores" {
  type    = number
  default = 28
}

variable "gpu_memory" {
  type    = number
  default = 100
}

variable "gpu_count" {
  type        = number
  description = "Количество GPU на узел (gpu-standard-v3: 1x A100)"
  default     = 1
}

variable "gpu_disk_size" {
  type    = number
  default = 200
}

variable "gpu_nodes" {
  type    = number
  default = 1
}

variable "gpu_zone" {
  type    = string
  default = "ru-central1-a"
}

variable "registry_id" {
  type    = string
  default = ""
}

variable "ssh_public_key" {
  type        = string
  description = "Публичный SSH-ключ для доступа к узлам"
  sensitive   = true
}
