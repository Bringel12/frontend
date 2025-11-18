import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from openai import OpenAI
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, select
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# --- Configuração MySQL com aiomysql ---
DB_USER = "root"
DB_PASSWORD = "Bringel@12"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "chat_db"

# Usando aiomysql (100% Python)
DATABASE_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    user_message: Mapped[str] = mapped_column(String(1000))
    ai_message: Mapped[str] = mapped_column(String(4000))


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando a aplicação e criando tabelas MySQL...")
    await create_db_and_tables()
    yield
    print("Desligando a aplicação...")

app = FastAPI(lifespan=lifespan)


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "SUA_CHAVE_OPENAI"))

# --- Dependência para sessão do DB ---
async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class Message(BaseModel):
    text: str

class ConversationRead(BaseModel):
    id: int
    user_message: str
    ai_message: str
    timestamp: datetime
    class Config:
        orm_mode = True


@app.post("/chat")
async def chat(message: Message, db: AsyncSession = Depends(get_db_session)):
    user_input = message.text
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}]
        )
        ai_reply = response.choices[0].message.content

        db_conversation = Conversation(user_message=user_input, ai_message=ai_reply)
        db.add(db_conversation)
        await db.commit()
        await db.refresh(db_conversation)

        return {"reply": ai_reply}

    except Exception as e:
        return {"error": str(e)}

@app.get("/conversas", response_model=list[ConversationRead])
async def get_conversas(db: AsyncSession = Depends(get_db_session), skip: int = 0, limit: int = 20):
    stmt = select(Conversation).order_by(Conversation.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    conversations = result.scalars().all()
    return conversations
