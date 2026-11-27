variable "project" {
  type = string
}

variable "environment" {
  type = string
}

variable "folder_id" {
  type = string
}

variable "kms_key_id" {
  type        = string
  description = "KMS-ключ из модуля kubernetes (пусто = AES256)"
  default     = ""
}

variable "state_bucket_max_size_gb" {
  type    = number
  default = 10
}

variable "dvc_expiration_days" {
  type        = number
  description = "Срок жизни объектов DVC (0 = бессрочно)"
  default     = 180
}

variable "labels" {
  type    = map(string)
  default = {}
}
