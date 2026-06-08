import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";
const AVATARS = ["🧑‍💻", "👩‍🎓", "🧑‍🎓", "👨‍💻", "🧠", "🦊", "🐼", "🚀"];

export default function App() {
  // ── Load from localStorage on first render ──────────────────────────
  const [page, setPage] = useState(() => localStorage.getItem("page") || "login");
  const [profile, setProfile] = useState(() => {
    const saved = localStorage.getItem("profile");
    return saved ? JSON.parse(saved) : { name: "", avatar: AVATARS[0] };
  });
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("messages");
    return saved ? JSON.parse(saved) : [];
  });

  const [nameInput, setNameInput] = useState("");
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0]);
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  // ── Save to localStorage whenever these change ───────────────────────
  useEffect(() => localStorage.setItem("page", page), [page]);
  useEffect(() => localStorage.setItem("profile", JSON.stringify(profile)), [profile]);
  useEffect(() => localStorage.setItem("messages", JSON.stringify(messages)), [messages]);

  // ── Load documents from backend on mount ─────────────────────────────
  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(res => setDocuments(res.data.documents))
      .catch(() => {});
  }, []);

  // ── Stats ─────────────────────────────────────────────────────────────
  const totalQuestions = messages.filter(m => m.role === "user").length;
  const successAnswers = messages.filter(m => m.role === "bot" && !m.error).length;
  const accuracy = totalQuestions === 0
    ? "0%"
    : `${Math.round((successAnswers / totalQuestions) * 100)}%`;

  // ── Login ─────────────────────────────────────────────────────────────
  function handleLogin() {
    if (!nameInput.trim()) return;
    const p = { name: nameInput.trim(), avatar: selectedAvatar };
    setProfile(p);
    setPage("home");
  }

  // ── Logout ────────────────────────────────────────────────────────────
  function handleLogout() {
    localStorage.clear();
    setMessages([]);
    setDocuments([]);
    setProfile({ name: "", avatar: AVATARS[0] });
    setPage("login");
  }

  // ── Upload ────────────────────────────────────────────────────────────
  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      await axios.post(`${API}/upload`, formData);
      const res = await axios.get(`${API}/documents`);
      setDocuments(res.data.documents);
      alert(`✅ ${file.name} uploaded!`);
    } catch (err) {
      alert("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  }

  // ── Delete Document ───────────────────────────────────────────────────
  async function handleDelete(filename) {
    try {
      await axios.delete(`${API}/documents/${encodeURIComponent(filename)}`);
      const res = await axios.get(`${API}/documents`);
      setDocuments(res.data.documents);
    } catch (err) {
      alert("Could not delete document.");
    }
  }

  // ── Chat ──────────────────────────────────────────────────────────────
  async function handleAsk() {
    if (!question.trim()) return;
    const newMessages = [...messages, { role: "user", text: question }];
    setMessages(newMessages);
    setQuestion("");
    setLoading(true);
    try {
      const res = await axios.post(`${API}/chat`, { question });
      setMessages([...newMessages, {
        role: "bot",
        text: res.data.answer,
        sources: res.data.sources,
        error: false
      }]);
    } catch (err) {
      setMessages([...newMessages, {
        role: "bot",
        text: "❌ Error getting answer. Is the backend running?",
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  // ── LOGIN PAGE ────────────────────────────────────────────────────────
  if (page === "login") return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">🧠</div>
        <h1 className="login-title">doclearn</h1>
        <p className="login-sub">your personal study assistant</p>
        <div className="avatar-picker">
          {AVATARS.map(a => (
            <button
              key={a}
              className={`avatar-btn ${selectedAvatar === a ? "selected" : ""}`}
              onClick={() => setSelectedAvatar(a)}
            >{a}</button>
          ))}
        </div>
        <input
          className="login-input"
          placeholder="what's your name?"
          value={nameInput}
          onChange={e => setNameInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleLogin()}
        />
        <button className="login-btn" onClick={handleLogin}>let's go →</button>
      </div>
    </div>
  );

  // ── PROFILE PAGE ──────────────────────────────────────────────────────
  if (page === "profile") return (
    <div className="app">
      <div className="navbar">
        <div className="navbar-brand">🧠 doclearn</div>
        <div className="navbar-right">
          <button className="nav-text-btn" onClick={() => setPage("home")}>← back</button>
          <button className="nav-text-btn logout" onClick={handleLogout}>logout</button>
        </div>
      </div>
      <div className="profile-page">
        <div className="profile-avatar-big">{profile.avatar}</div>
        <div className="profile-name">{profile.name}</div>
        <div className="profile-tag">learner</div>
        <div className="profile-stats">
          <div className="profile-stat">
            <div className="profile-stat-num">{documents.length}</div>
            <div className="profile-stat-label">docs uploaded</div>
          </div>
          <div className="profile-stat">
            <div className="profile-stat-num">{totalQuestions}</div>
            <div className="profile-stat-label">questions asked</div>
          </div>
          <div className="profile-stat">
            <div className="profile-stat-num">{accuracy}</div>
            <div className="profile-stat-label">accuracy rate</div>
          </div>
        </div>
        <div className="profile-docs">
          <div className="profile-docs-title">📚 your documents</div>
          {documents.length === 0
            ? <p className="empty-docs">no docs uploaded yet</p>
            : documents.map((doc, i) => (
              <div key={i} className="profile-doc-item">
                📄 {doc}
                <span className="doc-badge">ready</span>
                <button className="delete-btn" onClick={() => handleDelete(doc)}>🗑️</button>
              </div>
            ))
          }
        </div>
        <button className="clear-history-btn" onClick={() => {
          setMessages([]);
          localStorage.removeItem("messages");
        }}>
          🗑️ clear chat history
        </button>
      </div>
    </div>
  );

  // ── HOME PAGE ─────────────────────────────────────────────────────────
  return (
    <div className="app">
      <div className="navbar">
        <div className="navbar-brand">🧠 doclearn</div>
        <div className="navbar-right">
          <button className="profile-pill" onClick={() => setPage("profile")}>
            {profile.avatar} {profile.name}
          </button>
        </div>
      </div>

      <div className="hero">
        <div className="hero-top">
          <div className="hero-title">hey {profile.name}, welcome back!</div>
          <div className="hero-badge">you're on a roll</div>
        </div>
        <div className="hero-sub">
          drop a pdf and ask it anything, lowkey the smartest way to study
        </div>
      </div>

      <div className="stats-row">
        <div className="stat">
          <div className="stat-num">{documents.length}</div>
          <div className="stat-label">docs uploaded</div>
        </div>
        <div className="stat">
          <div className="stat-num">{totalQuestions}</div>
          <div className="stat-label">questions asked</div>
        </div>
        <div className="stat">
          <div className="stat-num">{accuracy}</div>
          <div className="stat-label">accuracy rate</div>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <div className="card-label">📂 drop your pdf here</div>
          <label className="upload-btn">
            {uploading ? "uploading..." : "📄 upload pdf"}
            <input
              type="file"
              accept=".pdf"
              onChange={handleUpload}
              disabled={uploading}
              style={{ display: "none" }}
            />
          </label>
          <div className="upload-hint">pdf files only</div>
        </div>

        <div className="card">
          <div className="card-label">📚 your documents</div>
          {documents.length === 0
            ? <p className="empty-docs">no docs yet, upload one!</p>
            : <ul className="doc-list">
              {documents.map((doc, i) => (
                <li key={i} className="doc-item">
                  <span className="doc-icon">📄</span>
                  <span className="doc-name">{doc}</span>
                  <span className="doc-badge">ready</span>
                  <button className="delete-btn" onClick={() => handleDelete(doc)}>🗑️</button>
                </li>
              ))}
            </ul>
          }
        </div>
      </div>

      <div className="chat-card">
        <div className="chat-label">💬 ask your document anything</div>
        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-chat">upload a pdf and ask something!</p>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <p>{msg.text}</p>
              {msg.sources && (
                <div className="sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-tag">
                      {s.filename} p.{s.page}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message bot"><p>thinking...</p></div>
          )}
        </div>
        <div className="input-row">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="ask something about your documents..."
            rows={2}
          />
          <button
            className="ask-btn"
            onClick={handleAsk}
            disabled={loading || !question.trim()}
          >
            {loading ? "..." : "ask"}
          </button>
        </div>
      </div>
    </div>
  );
}