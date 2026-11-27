output "network_id" {
  description = "ID VPC-сети"
  value       = yandex_vpc_network.this.id
}

output "public_subnet_ids" {
  description = "ID подсетей по зонам"
  value       = { for zone, s in yandex_vpc_subnet.public : zone => s.id }
}

output "subnet_id_by_zone" {
  description = "zone -> subnet_id (для master_location и node groups)"
  value       = { for key, cfg in var.zones : cfg.zone => yandex_vpc_subnet.public[key].id }
}

output "k8s_security_group_id" {
  value = yandex_vpc_security_group.k8s.id
}

output "nodes_security_group_id" {
  value = yandex_vpc_security_group.nodes.id
}

output "nat_route_table_id" {
  value = yandex_vpc_route_table.nat.id
}
