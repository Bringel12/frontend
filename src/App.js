import { useEffect, useRef, useState } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sources, setSources] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');
  const assistantIndexRef = useRef(-1);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/conversations`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      console.error('Erro ao carregar histórico', err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      fetchHistory();
    }
  }, [isStreaming]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!input.trim() || isStreaming) {
      return;
    }

    const question = input.trim();
    setInput('');
    setError('');
    setSources([]);
    setIsStreaming(true);

    setMessages((prev) => {
      assistantIndexRef.current = prev.length + 1;
      return [
        ...prev,
        { role: 'user', content: question },
        { role: 'assistant', content: '' }
      ];
    });

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: question })
      });

      if (!response.ok || !response.body) {
        throw new Error('Não foi possível iniciar o streaming');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          if (rawEvent.startsWith('data:')) {
            const payloadText = rawEvent.replace(/^data:\s*/, '');
            if (payloadText !== '[DONE]') {
              try {
                const payload = JSON.parse(payloadText);
                handleStreamPayload(payload);
              } catch (streamError) {
                console.error('Falha ao interpretar evento SSE', streamError);
              }
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
      }
    } catch (err) {
      console.error(err);
      setError('Falha ao conversar com o assistente.');
    } finally {
      setIsStreaming(false);
    }
  };

  const handleStreamPayload = (payload) => {
    if (payload.type === 'context') {
      setSources(payload.sources || []);
      return;
    }
    if (payload.type === 'token') {
      setMessages((prev) =>
        prev.map((message, index) =>
          index === assistantIndexRef.current
            ? { ...message, content: `${message.content}${payload.delta}` }
            : message
        )
      );
      return;
    }
    if (payload.type === 'done') {
      setIsStreaming(false);
    }
  };

  const handleSampleQuestion = (text) => {
    setInput(text);
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Stack: FastAPI · SQL · React</p>
          <h1>Assistente RAG com busca em SQL</h1>
          <p className="subtitle">
            Faça uma pergunta. O backend consulta o banco SQL, gera embeddings, cria um contexto e o LLM responde em tempo real.
          </p>
          <div className="sample-questions">
            {['Como o RAG funciona?', 'O que preciso para usar SQL com FastAPI?', 'Por que streaming melhora a UX?'].map((item) => (
              <button key={item} className="sample-chip" onClick={() => handleSampleQuestion(item)}>
                {item}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="chat-layout">
        <section className="chat-panel">
          <div className="chat-window">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Envie a primeira pergunta para ver o RAG em ação.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <span className="role">{message.role === 'user' ? 'Você' : 'Assistente'}</span>
                <p>{message.content}</p>
              </div>
            ))}
          </div>

          <form className="chat-input" onSubmit={handleSubmit}>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Pergunte algo sobre o projeto RAG"
              disabled={isStreaming}
            />
            <button type="submit" disabled={isStreaming || !input.trim()}>
              {isStreaming ? 'Gerando...' : 'Enviar'}
            </button>
          </form>
          {error && <p className="error-text">{error}</p>}

          {sources.length > 0 && (
            <div className="sources-panel">
              <h2>Fontes usadas</h2>
              <ul>
                {sources.map((source) => (
                  <li key={source.title}>
                    <p className="source-title">{source.title}</p>
                    <p className="source-meta">{source.source}</p>
                    <p className="source-snippet">{source.snippet}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        <aside className="history-panel">
          <div className="history-header">
            <h2>Últimas conversas</h2>
            <p>Dados gravados no banco SQL</p>
          </div>
          <ul>
            {history.map((conversation) => (
              <li key={conversation.id}>
                <p className="history-question">{conversation.user_message}</p>
                <p className="history-answer">{conversation.ai_message.slice(0, 120)}...</p>
                <p className="history-meta">{conversation.context_titles}</p>
              </li>
            ))}
            {history.length === 0 && <p className="empty-history">Nenhuma conversa armazenada ainda.</p>}
          </ul>
        </aside>
      </main>
    </div>
  );
}

export default App;
