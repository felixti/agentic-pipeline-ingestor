# Plano de Deploy - Agentic Pipeline Ingestor no Dokploy

## 📋 Resumo do Projeto

| Componente | Versão/Detalhes |
|------------|-----------------|
| **Framework** | FastAPI (Python 3.11+) |
| **Porta** | 8000 |
| **Banco de Dados** | PostgreSQL 17 com pgvector |
| **Cache** | Redis 7 |
| **Container** | Multi-stage Dockerfile |

---

## 1. Lista de Serviços Necessários

### 1.1 Serviços Principais (Obrigatórios)

| Serviço | Imagem | Porta | Descrição |
|---------|--------|-------|-----------|
| **API** | Build do Dockerfile | 8000 | FastAPI principal |
| **PostgreSQL** | `pgvector/pgvector:pg17` | 5432 | Banco com extensão pgvector |
| **Redis** | `redis:7-alpine` | 6379 | Cache e fila de jobs |

### 1.2 Serviços Opcionais (Recomendados para Produção)

| Serviço | Imagem | Porta | Descrição |
|---------|--------|-------|-----------|
| **Worker** | Build do Dockerfile | - | Processamento em background |
| **OpenSearch** | `opensearchproject/opensearch:2.11.0` | 9200 | Logs de auditoria |

---

## 2. Variáveis de Ambiente Obrigatórias

### 2.1 Configuração da Aplicação

```bash
# Ambiente
DEBUG=false
ENV=production
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Segurança (GERAR NOVAS CHAVES!)
SECURITY_SECRET_KEY=<chave-secreta-aleatoria-32-caracteres>
SECURITY_ALGORITHM=HS256
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=30
SECURITY_API_KEY_HEADER=X-API-Key
SECURITY_RATE_LIMIT_DEFAULT=100
SECURITY_RATE_LIMIT_WINDOW=60
SECURITY_CORS_ORIGINS=https://seu-dominio.com,https://app.seu-dominio.com
```

### 2.2 Banco de Dados PostgreSQL

```bash
# URL de conexão (substituir pelos valores do Dokploy)
DB_URL=postgresql+asyncpg://<user>:<password>@<host>:<port>/<database>
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_ECHO=false
```

### 2.3 Redis

```bash
# URL de conexão (substituir pelos valores do Dokploy)
REDIS_URL=redis://<user>:<password>@<host>:<port>/0
REDIS_SSL=true  # Em produção, usar SSL
```

### 2.4 LLM Providers (Escolher pelo menos um)

```bash
# Azure OpenAI (Recomendado)
AZURE_OPENAI_API_BASE=https://<seu-recurso>.openai.azure.com
AZURE_OPENAI_API_KEY=<sua-chave-azure>

# OpenRouter (Fallback)
OPENROUTER_API_KEY=<sua-chave-openrouter>

# OpenAI (Alternativo)
OPENAI_API_KEY=<sua-chave-openai>

# Anthropic (Alternativo)
ANTHROPIC_API_KEY=<sua-chave-anthropic>
```

### 2.5 Azure AI Vision (Para OCR)

```bash
AZURE_AI_VISION_ENDPOINT=https://<seu-recurso>.cognitiveservices.azure.com
AZURE_AI_VISION_API_KEY=<sua-chave-vision>
```

### 2.6 Observabilidade (Opcional)

```bash
OTEL_SERVICE_NAME=pipeline-ingestor
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=production
OTEL_OTLP_ENDPOINT=
OTEL_LOG_LEVEL=INFO
OTEL_PROMETHEUS_ENABLED=true
OTEL_PROMETHEUS_PORT=9090
```

### 2.7 Configuração de Processamento

```bash
PROCESSING_MAX_FILE_SIZE_MB=100
PROCESSING_MAX_PAGES_PER_DOCUMENT=1000
PROCESSING_TEMP_DIR=/tmp/pipeline
PROCESSING_CLEANUP_TEMP_FILES=true
PROCESSING_DEFAULT_TIMEOUT_SECONDS=300
PROCESSING_MAX_RETRIES=3
```

---

## 3. Passos de Deploy no Dokploy

### 3.1 Preparação Inicial

1. **Acesse o Dashboard do Dokploy**
   - URL: `https://seu-dokploy.com`
   - Faça login com suas credenciais

2. **Crie um Novo Projeto**
   - Clique em "New Project"
   - Nome: `agentic-pipeline-ingestor`
   - Descrição: `Pipeline de ingestão de documentos com FastAPI`

### 3.2 Configuração do Banco de Dados PostgreSQL

1. **Criar Serviço PostgreSQL**
   - Vá em "Databases" → "New Database"
   - Tipo: `PostgreSQL`
   - Versão: `17`
   - Nome do banco: `pipeline`
   - Usuário: `pipeline_user`
   - Senha: Gerar senha forte

2. **Habilitar pgvector**
   - Após criar o banco, conecte via console ou cliente SQL
   - Execute:
     ```sql
     CREATE EXTENSION IF NOT EXISTS vector;
     CREATE EXTENSION IF NOT EXISTS pg_trgm;
     ```

3. **Anotar Credenciais**
   - Host: `<nome-do-servico>` (nome interno do serviço)
   - Porta: `5432`
   - Usuário: `pipeline_user`
   - Senha: `<senha-gerada>`
   - Banco: `pipeline`

### 3.3 Configuração do Redis

1. **Criar Serviço Redis**
   - Vá em "Services" → "New Service"
   - Tipo: `Redis`
   - Versão: `7-alpine`
   - Nome: `pipeline-redis`

2. **Configurar Redis**
   - Adicione variável: `REDIS_PASSWORD=<senha-forte>`
   - Comando: `redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru --requirepass $REDIS_PASSWORD`

3. **Anotar Credenciais**
   - Host: `pipeline-redis`
   - Porta: `6379`
   - Senha: `<senha-configurada>`

### 3.4 Deploy da Aplicação FastAPI

1. **Criar Aplicação**
   - Vá em "Applications" → "New Application"
   - Nome: `pipeline-api`
   - Tipo: `Docker`

2. **Configurar Build**
   - **Source Type**: `Git`
   - **Repository**: URL do seu repositório Git
   - **Branch**: `main` (ou sua branch de produção)
   - **Dockerfile Path**: `Dockerfile` (raiz do projeto)
   - **Context Path**: `.`

3. **Configurar Portas**
   - Porta exposta: `8000`
   - Protocolo: `HTTP`

4. **Configurar Health Check**
   - Path: `/health/live`
   - Intervalo: `30s`
   - Timeout: `10s`
   - Retries: `3`

5. **Adicionar Variáveis de Ambiente**
   - Adicione todas as variáveis da seção 2
   - Para `DB_URL` e `REDIS_URL`, use os valores internos do Dokploy

6. **Configurar Recursos (Recomendado)**
   - CPU: `1-2 cores`
   - Memória: `2-4 GB`
   - Disco: `10-20 GB` (para uploads temporários)

### 3.5 Deploy do Worker (Opcional mas Recomendado)

1. **Criar Aplicação Worker**
   - Nome: `pipeline-worker`
   - Tipo: `Docker`
   - Mesmo repositório e branch

2. **Configurar Comando**
   - Comando de inicialização: `python -m src.worker.main`
   - Ou no Dockerfile, use: `CMD ["python", "-m", "src.worker.main"]`

3. **Configurar Variáveis de Ambiente**
   - Copiar as mesmas variáveis da API
   - Adicionar:
     ```bash
     WORKER_POLL_INTERVAL=5.0
     WORKER_MAX_CONCURRENT=3
     ```

4. **Configurar Réplicas**
   - Mínimo: `2` (para alta disponibilidade)
   - Máximo: `5` (baseado na carga)

### 3.6 Configuração de Domínio

1. **Adicionar Domínio na API**
   - Vá em "Applications" → `pipeline-api` → "Domains"
   - Clique em "Add Domain"
   - Domínio: `api.seu-dominio.com`
   - HTTPS: Ativar (Let's Encrypt)

2. **Configurar DNS**
   - No seu provedor DNS, crie um registro A:
     - Nome: `api`
     - Valor: `<IP-do-servidor-Dokploy>`

3. **Verificar SSL**
   - O Dokploy gerencia automaticamente certificados Let's Encrypt
   - Aguarde alguns minutos para propagação DNS

### 3.7 Configuração de Volumes (Persistência)

1. **Criar Volume para Uploads**
   - Vá em "Volumes" → "New Volume"
   - Nome: `pipeline-uploads`
   - Tamanho: `50 GB` (ajustar conforme necessidade)
   - Mount Path: `/tmp/pipeline`

2. **Associar Volume à API**
   - Na aplicação `pipeline-api`, adicione o volume
   - Mount Path: `/tmp/pipeline`

3. **Associar Volume ao Worker**
   - Na aplicação `pipeline-worker`, adicione o mesmo volume
   - Mount Path: `/tmp/pipeline`

---

## 4. Configuração de Domínio Completa

### 4.1 Estrutura de Domínios Recomendada

```
api.seu-dominio.com          → API FastAPI (porta 8000)
docs.seu-dominio.com         → Documentação Swagger (opcional)
```

### 4.2 Configuração de CORS

Adicione seus domínios na variável `SECURITY_CORS_ORIGINS`:

```bash
SECURITY_CORS_ORIGINS=https://seu-frontend.com,https://app.seu-dominio.com
```

### 4.3 Configuração de Rate Limiting

```bash
SECURITY_RATE_LIMIT_DEFAULT=100        # Requisições por janela
SECURITY_RATE_LIMIT_WINDOW=60          # Janela em segundos
```

---

## 5. Verificação Pós-Deploy

### 5.1 Health Checks

Execute os seguintes comandos para verificar o deploy:

```bash
# Health check básico
curl https://api.seu-dominio.com/health/live

# Health check completo
curl https://api.seu-dominio.com/health

# Verificar documentação
curl https://api.seu-dominio.com/docs

# Verificar OpenAPI spec
curl https://api.seu-dominio.com/api/v1/openapi.yaml
```

### 5.2 Testar Funcionalidades

```bash
# Testar autenticação
curl -X POST https://api.seu-dominio.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"senha"}'

# Testar upload de arquivo
curl -X POST https://api.seu-dominio.com/api/v1/upload \
  -H "X-API-Key: sua-api-key" \
  -F "files=@documento.pdf"

# Testar busca semântica
curl -X POST https://api.seu-dominio.com/api/v1/search/semantic \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua-api-key" \
  -d '{"query_embedding": [0.023, -0.045, ...], "top_k": 10}'
```

---

## 6. Troubleshooting

### 6.1 Problemas Comuns

| Problema | Causa Provável | Solução |
|----------|----------------|---------|
| API não inicia | Variáveis de ambiente faltando | Verificar todas as env vars obrigatórias |
| Erro de conexão DB | URL de conexão incorreta | Verificar `DB_URL` com host interno do Dokploy |
| Erro de conexão Redis | Redis não acessível | Verificar `REDIS_URL` e se o serviço está rodando |
| pgvector não funciona | Extensão não habilitada | Conectar no banco e executar `CREATE EXTENSION vector;` |
| Worker não processa jobs | Redis não configurado | Verificar conexão Redis e variáveis do worker |

### 6.2 Logs e Monitoramento

```bash
# Ver logs da API no Dokploy
# Applications → pipeline-api → Logs

# Ver logs do Worker
# Applications → pipeline-worker → Logs

# Ver logs do PostgreSQL
# Databases → <nome-do-banco> → Logs
```

---

## 7. Checklist de Deploy

- [ ] Criar projeto no Dokploy
- [ ] Configurar PostgreSQL 17 com pgvector
- [ ] Configurar Redis 7
- [ ] Fazer deploy da aplicação FastAPI
- [ ] Configurar variáveis de ambiente
- [ ] Configurar domínio com HTTPS
- [ ] Configurar volumes persistentes
- [ ] Fazer deploy do Worker (opcional)
- [ ] Executar health checks
- [ ] Testar endpoints principais
- [ ] Verificar logs de erro
- [ ] Configurar monitoramento (opcional)

---

## 8. Comandos Úteis

### 8.1 Migrações do Banco (Se necessário)

```bash
# Executar dentro do container da API
alembic upgrade head
```

### 8.2 Backup do Banco

```bash
# No Dokploy, use a funcionalidade de backup
# ou execute manualmente:
pg_dump -h <host> -U <user> -d pipeline > backup.sql
```

### 8.3 Escalar Workers

```bash
# No Dokploy, ajuste o número de réplicas
# Applications → pipeline-worker → Settings → Replicas
```

---

## 9. Recursos Adicionais

- **Documentação FastAPI**: https://fastapi.tiangolo.com/
- **Documentação pgvector**: https://github.com/pgvector/pgvector
- **Documentação Dokploy**: https://docs.dokploy.com/
- **OpenAPI Spec**: Disponível em `/api/v1/openapi.yaml`

---

## 10. Notas de Segurança

⚠️ **IMPORTANTE**:

1. **Nunca** commite o arquivo `.env` com credenciais reais
2. **Sempre** use HTTPS em produção
3. **Gere** chaves secretas fortes para `SECURITY_SECRET_KEY`
4. **Configure** CORS apenas para domínios confiáveis
5. **Monitore** logs de acesso regularmente
6. **Faça** backups automáticos do banco de dados
7. **Atualize** as dependências regularmente

---

*Plano gerado em: 2026-02-23*
*Versão: 1.0*
