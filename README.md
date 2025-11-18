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

## Como executar

### 0. Variáveis de ambiente

Copie o arquivo `.env.example` para `.env` (backend) e `.env.local` (frontend) e ajuste os valores:

- `DATABASE_URL`: string de conexão SQLAlchemy.
- `OPENAI_API_KEY`: chave da OpenAI (mantém o fallback local caso fique vazio).
- `OPENAI_MODEL`: modelo usado na API (ex.: `gpt-4o-mini`).
- `REACT_APP_API_URL`: URL do backend consumido pelo React.

Sem a chave de IA o backend continua respondendo, mas o `healthcheck` indicará o modo `local-fallback`.

### 1. Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Variáveis úteis:

- `DATABASE_URL`: URL compatível com SQLAlchemy (ex.: `sqlite:///./rag_chat.db`)
- `OPENAI_API_KEY`: usa GPT-4o-mini via API; sem a chave, o fallback local mantém as respostas

### 2. Frontend

```bash
npm install
npm start
```

A aplicação React conversa com `http://localhost:8000` por padrão. Altere com
`REACT_APP_API_URL` caso necessário. O projeto usa React 18.2 (compatível com CRA 5).

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
