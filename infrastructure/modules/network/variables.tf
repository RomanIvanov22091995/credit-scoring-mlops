variable "project" {
  description = "Имя проекта (для нейминга ресурсов)"
  type        = string
}

variable "environment" {
  description = "Окружение: staging | production"
  type        = string

  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment должен быть staging или production."
  }
}

variable "zones" {
  description = "Зоны доступности и CIDR-подсети"
  type = map(object({
    zone = string
    cidr = string
  }))
  default = {
    a = { zone = "ru-central1-a", cidr = "10.10.0.0/20" }
    b = { zone = "ru-central1-b", cidr = "10.10.16.0/20" }
    d = { zone = "ru-central1-d", cidr = "10.10.32.0/20" }
  }
}

variable "admin_cidrs" {
  description = "CIDR, из которых разрешён доступ к Kubernetes API"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "labels" {
  description = "Метки ресурсов"
  type        = map(string)
  default     = {}
}
