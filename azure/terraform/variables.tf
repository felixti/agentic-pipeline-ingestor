# Terraform variables for Azure deployment

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "pipeline"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

# AKS Variables
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "aks_node_count" {
  description = "Initial number of AKS nodes"
  type        = number
  default     = 3
}

variable "aks_node_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "aks_min_nodes" {
  description = "Minimum number of AKS nodes"
  type        = number
  default     = 3
}

variable "aks_max_nodes" {
  description = "Maximum number of AKS nodes"
  type        = number
  default     = 10
}

variable "enable_gpu_pool" {
  description = "Enable GPU node pool for ML workloads"
  type        = bool
  default     = false
}

variable "aks_admin_group_ids" {
  description = "Azure AD group IDs for AKS admins"
  type        = list(string)
  default     = []
}

# PostgreSQL Variables
variable "postgres_sku" {
  description = "PostgreSQL SKU"
  type        = string
  default     = "GP_Standard_D4s_v3"
}

variable "postgres_storage_mb" {
  description = "PostgreSQL storage in MB"
  type        = number
  default     = 524288  # 512 GB
}

variable "postgres_admin_username" {
  description = "PostgreSQL admin username"
  type        = string
  default     = "pipelineadmin"
}

variable "postgres_admin_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

# Redis Variables
variable "redis_sku" {
  description = "Redis SKU"
  type        = string
  default     = "Standard"
}

variable "redis_capacity" {
  description = "Redis capacity (0-6 for Standard/Premium)"
  type        = number
  default     = 1
}
