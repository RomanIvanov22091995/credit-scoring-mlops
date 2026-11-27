variable "yc_token" {
  type        = string
  description = "OAuth-токен Yandex Cloud (задаётся через TF_VAR_yc_token)"
  sensitive   = true
}

variable "cloud_id" {
  type        = string
  description = "ID облака"
}

variable "folder_id" {
  type        = string
  description = "ID каталога"
}

variable "project" {
  type    = string
  default = "credit-scoring"
}

variable "environment" {
  type    = string
  default = "staging"
}

variable "master_zone" {
  type    = string
  default = "ru-central1-a"
}

variable "admin_cidrs" {
  type        = list(string)
  description = "CIDR офисов/VPN с доступом к API кластера"
  default     = ["10.0.0.0/8"]
}

variable "registry_id" {
  type        = string
  description = "ID Yandex Container Registry"
  default     = ""
}

variable "ssh_public_key" {
  type        = string
  description = "Публичный SSH-ключ"
  sensitive   = true
}

variable "budget_amount" {
  type        = number
  description = "Месячный бюджет, руб. (алерты 80%/100%)"
  default     = 30000
}

variable "alert_email" {
  type    = string
  default = "mlops-team@example.com"
}
