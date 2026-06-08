import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function getInitials(name) {
  return name.split(" ").map(n => n[0]).join("").toUpperCase().slice(0, 2);
}

export default function App() {
  const [page, setPage] = useState(() => localStorage.getItem("page") || "login");
  const [profile, setProfile] = useState(() => {
    const saved = localStorage.getItem("profile");
    return saved ? JSON.parse(saved) : { name: "" };
  });
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("messages");
    return saved ? JSON.parse(saved) : [];
  });

  const [nameInput, setNameInput] = useState("");
  const [documents, setDocuments] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  useEffect(() => localStorage.setItem("page", page), [page]);
  useEffect(() => localStorage.setItem("profile", JSON.stringify(profile)), [profile]);
  useEffect(() => localStorage.setItem("messages", JSON.stringify(messages)), [messages]);

  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(res => setDocuments(res.data.documents))
      .catch(() => {});
  }, []);

  const totalQuestions = messages.filter(m => m.role === "user").length;
  const successAnswers = messages.filter(m => m.role === "bot" && !m.error).length;
  const accuracy = totalQuestions === 0
    ? "0%" : `${Math.round((successAnswers / totalQuestions) * 100)}%`;

  function handleLogin() {
    if (!nameInput.trim()) return;
    setProfile({ name: nameInput.trim() });
    setPage("home");
  }

  function handleLogout() {
    localStorage.clear();
    setMessages([]);
    setDocuments([]);
    setProfile({ name: "" });
    setPage("login");
  }

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
      alert(`${file.name} uploaded successfully.`);
    } catch {
      alert("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(filename) {
    try {
      await axios.delete(`${API}/documents/${encodeURIComponent(filename)}`);
      const res = await axios.get(`${API}/documents`);
      setDocuments(res.data.documents);
    } catch {
      alert("Could not delete document.");
    }
  }

  async function handleAsk() {
    if (!question.trim()) return;
    const newMessages = [...messages, { role: "user", text: question }];
    setMessages(newMessages);
    setQuestion("");
    setLoading(true);
    try {
      const res = await axios.post(`${API}/chat`, { question });
      setMessages([...newMessages, {
        role: "bot", text: res.data.answer,
        sources: res.data.sources, error: false
      }]);
    } catch {
      setMessages([...newMessages, {
        role: "bot",
        text: "Error getting answer. Is the backend running?",
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

  // ── LOGIN ──────────────────────────────────────────────────────────────
  if (page === "login") return (
    <div className="login-page">
      <div className="login-left">
        <div className="login-brand">doc<span>learn</span></div>
        <div className="login-headline">your documents,<br />answered instantly.</div>
        <div className="login-sub">upload any pdf and ask questions.<br />powered by ai, built for learners.</div>
        <div className="login-features">
          <div className="login-feature">
            <i className="ti ti-file-text" aria-hidden="true"></i>
            upload multiple pdfs
          </div>
          <div className="login-feature">
            <i className="ti ti-search" aria-hidden="true"></i>
            semantic search across docs
          </div>
          <div className="login-feature">
            <i className="ti ti-message-circle" aria-hidden="true"></i>
            ai-powered answers with sources
          </div>
        </div>
      </div>

      <div className="login-right">
        <div>
          <div className="form-title">get started</div>
          <div className="form-sub">enter your name to continue</div>
        </div>
        <div className="form-divider"></div>
        <div>
          <div className="form-label">your name</div>
          <input className="form-input" type="text"
            placeholder="e.g. taiba"
            value={nameInput}
            onChange={e => setNameInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleLogin()} />
        </div>
        <button className="form-btn" onClick={handleLogin}>
          continue <i className="ti ti-arrow-right" aria-hidden="true"></i>
        </button>
      </div>
    </div>
  );

  // ── PROFILE ────────────────────────────────────────────────────────────
  if (page === "profile") return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-brand">doc<span>learn</span></div>
        <div className="sidebar-divider"></div>
        <div className="sidebar-spacer"></div>
        <button className="sidebar-back" onClick={() => setPage("home")}>
          <i className="ti ti-arrow-left" aria-hidden="true"></i> back
        </button>
      </div>

      <div className="main">
        <div className="topbar">
          <div className="topbar-title">profile</div>
          <button className="logout-btn" onClick={handleLogout}>
            <i className="ti ti-logout" aria-hidden="true"></i> logout
          </button>
        </div>

        <div className="profile-body">
          <div className="profile-initials-big">
            {getInitials(profile.name)}
          </div>
          <div className="profile-name">{profile.name}</div>
          <div className="profile-role">learner</div>

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
            <div className="profile-docs-label">documents</div>
            {documents.length === 0
              ? <p className="empty-docs">no documents uploaded yet</p>
              : documents.map((doc, i) => (
                <div key={i} className="profile-doc-item">
                  <i className="ti ti-file-text" aria-hidden="true"></i>
                  <span>{doc}</span>
                  <button className="delete-btn" onClick={() => handleDelete(doc)}>
                    <i className="ti ti-x" aria-hidden="true"></i>
                  </button>
                </div>
              ))}
          </div>

          <button className="clear-btn" onClick={() => {
            setMessages([]);
            localStorage.removeItem("messages");
          }}>
            clear chat history
          </button>
        </div>
      </div>
    </div>
  );

  // ── HOME ───────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-brand">doc<span>learn</span></div>
        <div className="sidebar-divider"></div>

        <label className="upload-area">
          <i className="ti ti-upload" aria-hidden="true" style={{fontSize:"18px", color:"#7A9CC4"}}></i>
          <span className="upload-area-label">upload pdf</span>
          <span className="upload-btn-small">
            {uploading ? "uploading..." : "choose file"}
          </span>
          <input type="file" accept=".pdf" onChange={handleUpload}
            disabled={uploading} style={{display:"none"}} />
        </label>

        <div className="doc-section">
          <div className="doc-label">documents</div>
          {documents.length === 0
            ? <p className="empty-docs">no documents yet</p>
            : <ul className="doc-list">
              {documents.map((doc, i) => (
                <li key={i} className="doc-item">
                  <i className="ti ti-file-text doc-icon" aria-hidden="true"></i>
                  <span className="doc-name">{doc}</span>
                  <button className="delete-btn" onClick={() => handleDelete(doc)}>
                    <i className="ti ti-x" aria-hidden="true"></i>
                  </button>
                </li>
              ))}
            </ul>
          }
        </div>

        <button className="profile-row" onClick={() => setPage("profile")}>
          <div className="avatar-initials">{getInitials(profile.name)}</div>
          <span className="profile-name-small">{profile.name}</span>
          <i className="ti ti-chevron-right" aria-hidden="true"
            style={{fontSize:"12px", color:"#4A6080", marginLeft:"auto"}}></i>
        </button>
      </div>

      <div className="main">
        <div className="topbar">
          <div className="topbar-title">conversation</div>
          <div className="topbar-stats">
            <div className="stat">
              <div className="stat-num">{documents.length}</div>
              <div className="stat-label">docs</div>
            </div>
            <div className="stat">
              <div className="stat-num">{totalQuestions}</div>
              <div className="stat-label">asked</div>
            </div>
            <div className="stat">
              <div className="stat-num">{accuracy}</div>
              <div className="stat-label">accuracy</div>
            </div>
          </div>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-chat">upload a pdf and start asking questions.</p>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <p>{msg.text}</p>
              {msg.sources && (
                <div className="sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-tag">
                      {s.filename} · p.{s.page}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="message bot"><p>thinking...</p></div>}
        </div>

        <div className="input-row">
          <textarea value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="ask anything about your documents..."
            rows={2} />
          <button className="ask-btn" onClick={handleAsk}
            disabled={loading || !question.trim()}>
            {loading ? "..." : <><span>send</span> <i className="ti ti-arrow-right" aria-hidden="true"></i></>}
          </button>
        </div>
      </div>
    </div>
  );
}