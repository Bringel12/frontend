import json
import os
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from typing import AsyncGenerator, List, Sequence
from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy import DateTime, String, Text, select
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./rag_chat.db")


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text())
    source: Mapped[str] = mapped_column(String(200))
    tags: Mapped[str] = mapped_column(String(200), default="")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    user_message: Mapped[str] = mapped_column(Text())
    ai_message: Mapped[str] = mapped_column(Text())
    context_titles: Mapped[str] = mapped_column(String(500))


engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(engine, autoflush=False, autocommit=False, future=True)


def get_db_session() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class Message(BaseModel):
    text: str


class SourceDocument(BaseModel):
    title: str
    source: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceDocument]


class ConversationRead(BaseModel):
    id: int
    user_message: str
    ai_message: str
    timestamp: datetime
    context_titles: str

    class Config:
        from_attributes = True


def seed_payload() -> List[dict]:
    return [
        {
            "title": "Arquitetura do pipeline RAG",
            "content": (
                "RAG combina busca vetorial e modelos de linguagem. Primeiro, o sistema gera "
                "embeddings com Sentence Transformers, executa uma busca aproximada para trazer "
                "trechos relevantes e só então envia o contexto para o LLM escolhido."
            ),
            "source": "guia-rag.md",
            "tags": "rag,arquitetura"
        },
        {
            "title": "Conexão com bancos SQL",
            "content": (
                "Um backend FastAPI pode usar SQLAlchemy para conversar com diferentes bancos. "
                "A abordagem assíncrona funciona bem com SQLite (aiosqlite) e MySQL (aiomysql)."
            ),
            "source": "sql-best-practices.md",
            "tags": "sql,fastapi"
        },
        {
            "title": "Boas práticas para prompts",
            "content": (
                "Prompts efetivos delimitam o contexto com instruções claras, mencionam o tom desejado "
                "e lembram o modelo para citar a origem das informações usadas na resposta."
            ),
            "source": "prompt-engineering.md",
            "tags": "prompting"
        },
        {
            "title": "Streaming no frontend",
            "content": (
                "No React é comum usar fetch com ReadableStream ou EventSource para renderizar tokens à medida "
                "que chegam do backend. Isso melhora a percepção de velocidade do usuário."
            ),
            "source": "frontend-streaming.md",
            "tags": "frontend,ux"
        },
    ]


class EmbeddingBackend:
    def __init__(self) -> None:
        self.vocabulary: List[str] = []

    def embed_corpus(self, documents: Sequence[str]) -> List[List[float]]:
        if not documents:
            return []
        tokens = [self._tokenize(doc) for doc in documents]
        vocab = sorted({token for doc_tokens in tokens for token in doc_tokens})
        self.vocabulary = vocab
        return [self._vectorize(doc_tokens) for doc_tokens in tokens]

    def embed_query(self, query: str) -> List[float]:
        if not self.vocabulary:
            raise RuntimeError("Nenhum vocabulário foi definido para gerar embeddings.")
        return self._vectorize(self._tokenize(query))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text)
        return [token for token in cleaned.split() if token]

    def _vectorize(self, tokens: Sequence[str]) -> List[float]:
        counter = Counter(tokens)
        return [float(counter.get(term, 0)) for term in self.vocabulary]


class LocalLLM:
    def __init__(self) -> None:
        self.prefix = (
            "Segue uma resposta resumida e objetiva usando somente o contexto enviado."
        )

    async def generate(self, prompt: str) -> str:
        return self._fallback(prompt)

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        text = await self.generate(prompt)
        for chunk in self._chunk(text):
            yield chunk

    @staticmethod
    def _chunk(text: str) -> List[str]:
        size = 40
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _fallback(self, prompt: str) -> str:
        return f"{self.prefix}\n\n{prompt.split('Contexto:')[-1].strip()}"


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.local = LocalLLM()

    async def generate(self, prompt: str) -> str:
        if self.client is None:
            return await self.local.generate(prompt)

        def _call() -> str:
            completion = self.client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": "Você é um assistente especialista em RAG."},
                          {"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content

        return await run_in_threadpool(_call)

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        text = await self.generate(prompt)
        for chunk in LocalLLM._chunk(text):
            yield chunk


PROMPT_TEMPLATE = (
    "Você é um assistente técnico que responde em português. Use apenas os fatos a seguir.\n"
    "Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
)


class RAGPipeline:
    def __init__(self, embedder: EmbeddingBackend, llm: LLMClient) -> None:
        self.embedder = embedder
        self.llm = llm

    def retrieve_context(self, query: str, session: Session, top_k: int = 3):
        documents = session.execute(select(Document)).scalars().all()
        if not documents:
            raise HTTPException(status_code=500, detail="Nenhum documento disponível para busca.")

        doc_texts = [doc.content for doc in documents]
        doc_embeddings = self.embedder.embed_corpus(doc_texts)
        query_embedding = self.embedder.embed_query(query)

        if not doc_embeddings:
            raise HTTPException(status_code=500, detail="Falha ao calcular embeddings.")

        scores = self._cosine_similarity(query_embedding, doc_embeddings)
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        top_ranked = ranked[:top_k]
        sources = [
            SourceDocument(
                title=doc.title,
                source=doc.source,
                snippet=doc.content[:180] + ("..." if len(doc.content) > 180 else ""),
                score=float(score)
            )
            for doc, score in top_ranked
        ]
        context_text = "\n\n".join(
            f"{doc.title} ({doc.source}): {doc.content}" for doc, _ in top_ranked
        )
        return context_text, sources

    @staticmethod
    def _cosine_similarity(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> List[float]:
        def dot(a: Sequence[float], b: Sequence[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def norm(a: Sequence[float]) -> float:
            return sqrt(sum(x * x for x in a))

        vector_norm = norm(vector) or 1e-9
        scores: List[float] = []
        for row in matrix:
            row_norm = norm(row) or 1e-9
            scores.append(dot(row, vector) / (row_norm * vector_norm))
        return scores

    def build_prompt(self, context: str, question: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    async def generate_answer(self, prompt: str) -> str:
        return await self.llm.generate(prompt)

    async def stream_answer(self, prompt: str) -> AsyncGenerator[str, None]:
        async for chunk in self.llm.stream(prompt):
            yield chunk


embedding_backend = EmbeddingBackend()
llm_client = LLMClient()
rag_pipeline = RAGPipeline(embedding_backend, llm_client)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"]
)


async def create_db_and_seed() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        result = session.execute(select(Document))
        if result.scalars().first():
            return
        for payload in seed_payload():
            session.add(Document(**payload))
        session.commit()


@app.on_event("startup")
async def startup_event() -> None:
    await create_db_and_seed()


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/documents", response_model=List[SourceDocument])
async def list_documents(db: Session = Depends(get_db_session)):
    documents = db.execute(select(Document)).scalars().all()
    return [
        SourceDocument(
            title=doc.title,
            source=doc.source,
            snippet=doc.content[:200] + ("..." if len(doc.content) > 200 else ""),
            score=1.0
        )
        for doc in documents
    ]


@app.get("/conversations", response_model=List[ConversationRead])
async def list_conversations(db: Session = Depends(get_db_session)):
    result = db.execute(select(Conversation).order_by(Conversation.timestamp.desc()).limit(20))
    return result.scalars().all()


def persist_conversation(
    db: Session, user_message: str, ai_message: str, sources: List[SourceDocument]
) -> None:
    conversation = Conversation(
        user_message=user_message,
        ai_message=ai_message,
        context_titles=", ".join(source.title for source in sources)
    )
    db.add(conversation)
    db.commit()


@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message, db: Session = Depends(get_db_session)):
    context, sources = rag_pipeline.retrieve_context(message.text, db)
    prompt = rag_pipeline.build_prompt(context, message.text)
    answer = await rag_pipeline.generate_answer(prompt)
    persist_conversation(db, message.text, answer, sources)
    return ChatResponse(reply=answer, sources=sources)


def format_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(message: Message, db: Session = Depends(get_db_session)):
    context, sources = rag_pipeline.retrieve_context(message.text, db)
    prompt = rag_pipeline.build_prompt(context, message.text)
    answer_accumulator = {"text": ""}

    async def event_generator() -> AsyncGenerator[str, None]:
        yield format_sse({"type": "context", "sources": [s.model_dump() for s in sources]})
        async for chunk in rag_pipeline.stream_answer(prompt):
            answer_accumulator["text"] += chunk
            yield format_sse({"type": "token", "delta": chunk})
        persist_conversation(db, message.text, answer_accumulator["text"], sources)
        yield format_sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
