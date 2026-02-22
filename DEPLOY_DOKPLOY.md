# Plano de Deploy - Agentic Pipeline Ingestor no Dokploy

## 📋 Resumo do Projeto

| Item | Detalhe |
|------|---------|
| **Nome** | Agentic Pipeline Ingestor |
| **Framework** | FastAPI (Python 3.11+) |
| **Porta** | 8000 |
| **Banco de Dados** | PostgreSQL 17+ com pgvector |
| **Cache** | Redis 7+ |
| **Health Checks** | /health/live e /health/ready |

## 🏗️ Arquitetura de Deploy no Dokploy

### Serviços Necessários

1. **API (FastAPI)** - Porta 8000
2. **Worker** - Background tasks (opcional)
3. **PostgreSQL** - Com extensão pgvector
4. **Redis** - Cache e message broker
5. **OpenSearch** - Audit logs (opcional)

### Configuração no Dokploy

#### 1. Criar Projeto

```bash
# Via API
curl -X POST 'https://dokploy.felixtek.cloud/api/project.create' \
  -H 'x-api-key: SUA_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"name": "agentic-pipeline", "description": "Agentic Data Pipeline Ingestor"}'
```

#### 2. Banco de Dados PostgreSQL

No Dokploy, crie um PostgreSQL com pgvector:
- **Image**: `pgvector/pgvector:pg17`
- **Database**: `pipeline`
- **User**: `postgres`
- **Password**: (gerar senha segura)

#### 3. Redis

No Dokploy, crie um Redis:
- **Image**: `redis:7-alpine`
- **Command**: `redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru`

#### 4. Aplicação Principal (API)

**Tipo**: Docker
**Dockerfile**: Use o Dockerfile existente no projeto

**Variáveis de Ambiente Obrigatórias**:

```env
# Application
DEBUG=false
ENV=production
HOST=0.0.0.0
PORT=8000

# Database (substituir com valores do Dokploy)
DB_URL=postgresql+asyncpg://postgres:SENHA@postgres:5432/pipeline
DB_ECHO=false

# Redis (substituir com valores do Dokploy)
REDIS_URL=redis://redis:6379/0

# Security (gerar chave segura)
SECRET_KEY=chave-secreta-aleatoria-de-32-caracteres

# LLM Configuration (escolher um)
OPENROUTER_API_KEY=sua-chave-openrouter
# ou
AZURE_OPENAI_API_KEY=sua-chave-azure
AZURE_OPENAI_API_BASE=https://seu-recurso.openai.azure.com

# Azure AI Vision (para OCR - opcional)
AZURE_AI_VISION_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com
AZURE_AI_VISION_API_KEY=sua-chave-vision

# Cognee (opcional)
COGNEE_API_URL=http://cognee:8001
COGNEE_API_KEY=sua-chave-cognee

# Observability
OTEL_SERVICE_NAME=pipeline-api
OTEL_ENVIRONMENT=production
OTEL_LOG_LEVEL=INFO
```

#### 5. Worker (Opcional)

Criar segunda aplicação para o worker:
- **Mesma imagem** da API
- **Command**: `python -m src.worker.main`
- **Mesmas variáveis de ambiente**

#### 6. Domínio

Configurar domínio no Dokploy:
- **Domain**: `pipeline.felixtek.cloud` (ou outro subdomínio)
- **Port**: 8000
- **HTTPS**: Habilitado (Let's Encrypt)

## 📁 Estrutura de Arquivos para Deploy

```
agentic-pipeline-ingestor/
├── Dockerfile                 # ✅ Já existe
├── docker/
│   └── docker-compose.yml    # ✅ Referência
├── src/
│   ├── main.py               # Entry point da API
│   └── worker/
│       └── main.py           # Entry point do worker
├── config/
│   └── vector_store.yaml     # Configurações
├── migrations/               # Alembic migrations
├── pyproject.toml           # Dependências Python
└── README.md
```

## 🔧 Passos de Deploy

### 1. Preparação

```bash
# Clone o repositório
git clone https://github.com/felixti/agentic-pipeline-ingestor.git
cd agentic-pipeline-ingestor

# Verifique se está na branch correta
git branch
```

### 2. Build da Imagem

O Dokploy pode fazer o build automaticamente a partir do Dockerfile.

### 3. Configuração de Variáveis

1. Acesse o painel do Dokploy: https://dokploy.felixtek.cloud
2. Crie o projeto e as aplicações
3. Configure as variáveis de ambiente
4. Gere a `SECRET_KEY`: `openssl rand -hex 32`

### 4. Banco de Dados

```sql
-- O pgvector já vem instalado na imagem
-- Apenas crie as tabelas via migrations
```

### 5. Migrations

Execute as migrations no container da API:

```bash
# Via Dokploy terminal ou:
docker exec -it container-api alembic upgrade head
```

### 6. Health Checks

Configure no Dokploy:
- **Path**: `/health/live`
- **Port**: 8000
- **Interval**: 30s

## 🔒 Segurança

### Variáveis Sensíveis

- `SECRET_KEY` - Gerar chave aleatória forte
- `OPENROUTER_API_KEY` - Ou Azure OpenAI
- `DB_URL` - Com senha do PostgreSQL
- `AZURE_AI_VISION_API_KEY` - Se usar OCR

### Recomendações

1. **Nunca** commite o arquivo `.env`
2. Use secrets do Dokploy para variáveis sensíveis
3. Habilite HTTPS no domínio
4. Configure firewall para bloquear portas desnecessárias

## 📊 Monitoramento

### Endpoints de Health

- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

### Métricas (Opcional)

Configure Prometheus/Grafana via profiles no docker-compose.

## 🚀 Comandos Úteis

### Ver logs
```bash
# Via Dokploy UI ou:
docker logs -f container-api
```

### Restart
```bash
# Via Dokploy UI ou API
```

### Shell no container
```bash
docker exec -it container-api /bin/bash
```

## ⚠️ Considerações Importantes

1. **pgvector**: O Dokploy precisa suportar imagens PostgreSQL customizadas ou você precisa usar uma imagem com pgvector pré-instalado

2. **Worker**: O worker é opcional mas recomendado para processamento em background

3. **OpenSearch**: Opcional para audit logs. Se não usar, remova as variáveis relacionadas

4. **Volumes**: Configure volumes persistentes no Dokploy para:
   - Uploads de arquivos
   - Dados do PostgreSQL
   - Dados do Redis

5. **Recursos**: Monitore uso de CPU/memória e ajuste conforme necessário

## 📚 Documentação Adicional

- [README.md](./README.md) - Documentação do projeto
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Arquitetura detalhada
- [Dokploy Docs](https://docs.dokploy.com) - Documentação do Dokploy

---

**Status**: ✅ Pronto para deploy
**Última atualização**: 2026-02-23
