# Azure Deployment Guide

This directory contains Azure deployment configurations for the Agentic Data Pipeline Ingestor.

## Overview

The deployment includes:
- **AKS Cluster** - Kubernetes cluster for running API and Worker services
- **Azure DB for PostgreSQL** - Primary + Replica database
- **Azure Cache for Redis** - Session and cache storage
- **Azure Blob Storage** - File storage and queue storage
- **Azure Monitor** - Application Insights for metrics, logs, and traces
- **Azure Key Vault** - Secret management
- **Azure Container Registry** - Docker image registry

## Deployment Options

### 1. Azure ARM Template

Deploy using the Azure Portal, Azure CLI, or PowerShell:

```bash
# Login to Azure
az login

# Create resource group
az group create --name pipeline-ingestor-rg --location eastus

# Deploy using ARM template
az deployment group create \
  --resource-group pipeline-ingestor-rg \
  --template-file azuredeploy.json \
  --parameters environmentName=prod postgresAdminPassword=YourSecurePassword123!
```

### 2. Terraform

For infrastructure-as-code deployment:

```bash
# Initialize Terraform
cd terraform
terraform init

# Create tfvars file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Plan deployment
terraform plan

# Apply deployment
terraform apply

# Get outputs
terraform output
```

### 3. Kubernetes Manifests

Deploy to an existing AKS cluster:

```bash
# Configure kubectl
az aks get-credentials --name pipeline-prod-aks --resource-group pipeline-ingestor-rg

# Apply manifests
kubectl apply -f aks-deployment.yaml

# Verify deployment
kubectl get pods -n pipeline-ingestor
kubectl get svc -n pipeline-ingestor
```

## Configuration

### Required Secrets

Create the following secrets in Azure Key Vault or Kubernetes:

| Secret | Description |
|--------|-------------|
| `DB_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `SECRET_KEY` | JWT signing key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_AI_VISION_API_KEY` | Azure AI Vision API key |
| `COGNEE_API_KEY` | Cognee API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | App Insights connection |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | production | Environment name |
| `OTEL_ENABLED` | true | Enable OpenTelemetry |
| `OTEL_SERVICE_NAME` | pipeline-api | Service name for tracing |
| `WORKER_MAX_CONCURRENT` | 5 | Max concurrent worker tasks |

## Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

The deployment includes HPAs for automatic scaling:

**API Service:**
- Min replicas: 3
- Max replicas: 10
- Scale on CPU > 70% or Memory > 80%

**Worker Service:**
- Min replicas: 5
- Max replicas: 20
- Scale on queue depth or CPU > 75%

### Manual Scaling

```bash
# Scale API
kubectl scale deployment pipeline-api --replicas=5 -n pipeline-ingestor

# Scale Workers
kubectl scale deployment pipeline-worker --replicas=10 -n pipeline-ingestor
```

## Monitoring

### Azure Monitor

Access Application Insights metrics:
```bash
az monitor app-insights metrics show \
  --app pipeline-prod-appinsights \
  --metric requests/count \
  --interval PT1H
```

### Prometheus Metrics

If using Prometheus in-cluster:
```bash
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Open http://localhost:9090
```

### Logs

```bash
# View API logs
kubectl logs -f deployment/pipeline-api -n pipeline-ingestor

# View Worker logs
kubectl logs -f deployment/pipeline-worker -n pipeline-ingestor
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n pipeline-ingestor
kubectl describe pod <pod-name> -n pipeline-ingestor
```

### Health Checks

```bash
# API health
curl http://<external-ip>/health

# Readiness probe
curl http://<external-ip>/health/ready

# Liveness probe
curl http://<external-ip>/health/live

# Metrics
curl http://<external-ip>/metrics
```

### Common Issues

1. **ImagePullBackOff**: Check ACR credentials and image tags
2. **CrashLoopBackOff**: Check application logs and secrets
3. **Pending pods**: Check resource quotas and node capacity

## Cleanup

```bash
# Delete Kubernetes resources
kubectl delete -f aks-deployment.yaml

# Delete Terraform resources
cd terraform
terraform destroy

# Or delete resource group (removes everything)
az group delete --name pipeline-ingestor-rg --yes
```

## Security Best Practices

1. Use Azure AD Workload Identity for pod authentication
2. Enable Azure Policy for AKS
3. Use private endpoints for PostgreSQL and Redis
4. Enable NSG flow logs
5. Regularly rotate secrets in Key Vault
6. Enable Defender for Containers

## Cost Optimization

1. Use spot instances for worker nodes (if applicable)
2. Right-size VMs based on actual usage
3. Enable auto-shutdown for dev environments
4. Use reserved instances for production workloads
5. Monitor and optimize storage costs
