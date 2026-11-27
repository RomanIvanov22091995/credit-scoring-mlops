output "cluster_id" {
  value = yandex_kubernetes_cluster.this.id
}

output "cluster_name" {
  value = yandex_kubernetes_cluster.this.name
}

output "cluster_public_ip" {
  value = yandex_kubernetes_cluster.this.master[0].public_ip
}

output "cluster_ca_certificate" {
  value     = yandex_kubernetes_cluster.this.master[0].cluster_ca_certificate
  sensitive = true
}

output "cpu_node_group_id" {
  value = yandex_kubernetes_node_group.cpu.id
}

output "gpu_node_group_id" {
  value = try(yandex_kubernetes_node_group.gpu[0].id, null)
}

output "kms_key_id" {
  value = yandex_kms_symmetric_key.kms.id
}
