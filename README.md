# Assistente RAG com FastAPI + React

Aplicação full-stack que demonstra um fluxo Retrieval Augmented Generation (RAG) completo.
O backend FastAPI consulta documentos salvos em um banco SQL (SQLite por padrão), calcula
embeddings simples (bag-of-words), seleciona os trechos mais relevantes e gera uma resposta
usando um LLM remoto (OpenAI) ou um gerador local determinístico. O frontend em React
consome o endpoint de streaming e renderiza os tokens em tempo real.

## Tecnologias principais

- **Backend**: FastAPI + SQLAlchemy (sincrono) + pipeline de RAG (embeddings + retrieval + LLM)
- **Banco**: SQLite (padrão) – pode ser trocado configurando `DATABASE_URL`
- **Frontend**: React com Create React App
- **Testes**: Pytest (backend) e React Testing Library (frontend)

## Estrutura do repositório

Todo o código fica diretamente na raiz (`/workspace/frontend`). Em vez de duas pastas separadas
"backend" e "frontend", usamos os seguintes caminhos:

| Caminho | O que contém |
| --- | --- |
| `app.py` | Servidor FastAPI com o pipeline RAG (banco + embeddings + retrieval + LLM + SSE). |
| `requirements.txt` | Dependências Python usadas pelo backend. |
| `src/`, `public/`, `package.json` | Projeto React criado com Create React App. |
| `tests/` | Testes Pytest que validam a API. |
| `.env.example` | Modelo com todas as variáveis necessárias. |

Basta abrir um terminal na raiz para seguir os próximos passos.

## Como executar

### 0. Variáveis de ambiente

Copie o arquivo `.env.example` para `.env` (backend) e `.env.local` (frontend) e ajuste os valores:

- `DATABASE_URL`: string de conexão SQLAlchemy.
- `OPENAI_API_KEY`: chave da OpenAI (mantém o fallback local caso fique vazio).
- `OPENAI_MODEL`: modelo usado na API (ex.: `gpt-4o-mini`).
- `REACT_APP_API_URL`: URL do backend consumido pelo React.

Sem a chave de IA o backend continua respondendo, mas o `healthcheck` indicará o modo `local-fallback`.

### 1. Backend (FastAPI)

Execute todos os comandos a seguir dentro da pasta `/workspace/frontend`:

```bash
# criar o ambiente virtual
python -m venv .venv

# ativar o ambiente
source .venv/bin/activate

# instalar dependências
pip install -r requirements.txt

# iniciar o servidor na porta 8000
uvicorn app:app --reload
```

Enquanto o `uvicorn` estiver rodando ele já vai ler `DATABASE_URL`, `OPENAI_API_KEY` e demais
variáveis do `.env`. Deixe esse terminal aberto para acompanhar logs.

### 2. Frontend (React)

```bash
npm install
npm start
```

A aplicação React conversa com `http://localhost:8000` por padrão. Se o backend estiver em outra
URL, ajuste `REACT_APP_API_URL` no arquivo `.env.local`. É recomendável manter um segundo terminal
apenas para o frontend.

## Testes

```bash
pytest
npm test -- --watchAll=false
```

> **Nota**: em ambientes corporativos com bloqueio ao registry do npm é normal ver erros 403 ao executar `npm install`. Nesses casos
> configure o proxy/liberação antes de iniciar o frontend.

## Fluxo

1. Usuário envia pergunta no React
2. Backend busca documentos no SQL, gera embeddings e escolhe os melhores trechos
3. Contexto é enviado ao LLM que responde; o resultado é transmitido via SSE
4. Frontend renderiza os tokens e exibe as fontes usadas
5. Conversas ficam persistidas para auditoria
