import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import "./App.css";

// ── Constants ────────────────────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const CLASS_EMOJIS = {
  airplane:   "✈️",
  automobile: "🚗",
  bird:       "🐦",
  cat:        "🐱",
  deer:       "🦌",
  dog:        "🐶",
  frog:       "🐸",
  horse:      "🐴",
  ship:       "🚢",
  truck:      "🚛",
};

// ── Sub-components ────────────────────────────────────────────────────────────

function ConfidenceBar({ label, confidence, isTop }) {
  const pct = (confidence * 100).toFixed(1);
  return (
    <div className={`confidence-row ${isTop ? "top-prediction" : ""}`}>
      <span className="conf-label">
        {CLASS_EMOJIS[label] || "🖼️"} {label}
      </span>
      <div className="conf-bar-wrap">
        <div className="conf-bar" style={{ width: `${pct}%` }} />
      </div>
      <span className="conf-pct">{pct}%</span>
    </div>
  );
}

function ResultCard({ result }) {
  if (!result) return null;
  const { prediction, confidence, top_k, inference_time_ms } = result;

  return (
    <div className="result-card">
      <h2 className="result-title">
        {CLASS_EMOJIS[prediction] || "🖼️"} {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
      </h2>
      <p className="result-confidence">
        {(confidence * 100).toFixed(1)}% confidence
      </p>
      <div className="top-k">
        <h3>Top-5 Predictions</h3>
        {top_k.map((item, i) => (
          <ConfidenceBar
            key={item.label}
            label={item.label}
            confidence={item.confidence}
            isTop={i === 0}
          />
        ))}
      </div>
      <p className="inference-time">⚡ Inference time: {inference_time_ms} ms</p>
    </div>
  );
}

function ErrorBanner({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="error-banner" role="alert">
      <span>⚠️ {message}</span>
      <button className="dismiss-btn" onClick={onDismiss} aria-label="Dismiss">✕</button>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [preview, setPreview]   = useState(null);   // data URL
  const [file, setFile]         = useState(null);   // File object
  const [result, setResult]     = useState(null);   // API response
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  // ── Dropzone ────────────────────────────────────────────────────────────────
  const onDrop = useCallback((accepted, rejected) => {
    setError(null);
    setResult(null);

    if (rejected.length > 0) {
      setError("Only JPEG, PNG, or WebP images under 10 MB are accepted.");
      return;
    }
    if (accepted.length === 0) return;

    const f = accepted[0];
    setFile(f);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [], "image/png": [], "image/webp": [] },
    maxSize: 10 * 1024 * 1024,   // 10 MB
    multiple: false,
  });

  // ── Predict ──────────────────────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30_000,
      });
      setResult(response.data);
    } catch (err) {
      if (err.response) {
        setError(err.response.data?.detail || `Server error (${err.response.status})`);
      } else if (err.request) {
        setError("Cannot reach the server. Is the backend running?");
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = () => {
    setPreview(null);
    setFile(null);
    setResult(null);
    setError(null);
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <h1 className="app-title">🔍 CIFAR-10 Image Classifier</h1>
          <p className="app-subtitle">
            Upload an image and let the CNN identify it across 10 categories
          </p>
          <div className="class-pills">
            {Object.entries(CLASS_EMOJIS).map(([name, emoji]) => (
              <span key={name} className="pill">{emoji} {name}</span>
            ))}
          </div>
        </div>
      </header>

      <main className="app-main">
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        <div className="content-grid">
          {/* Upload Panel */}
          <section className="upload-panel">
            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? "drag-over" : ""} ${preview ? "has-image" : ""}`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <img src={preview} alt="Uploaded preview" className="preview-img" />
              ) : (
                <div className="dropzone-placeholder">
                  <span className="drop-icon">📁</span>
                  <p className="drop-primary">
                    {isDragActive ? "Drop it here …" : "Drag & drop an image here"}
                  </p>
                  <p className="drop-secondary">or click to browse</p>
                  <p className="drop-hint">JPEG · PNG · WebP · max 10 MB</p>
                </div>
              )}
            </div>

            <div className="action-row">
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={!file || loading}
              >
                {loading ? (
                  <><span className="spinner" /> Classifying…</>
                ) : (
                  "🚀 Classify Image"
                )}
              </button>
              {(preview || result) && (
                <button className="btn btn-secondary" onClick={handleReset}>
                  🔄 Reset
                </button>
              )}
            </div>

            {file && (
              <p className="file-info">
                📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </section>

          {/* Results Panel */}
          <section className="results-panel">
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="results-placeholder">
                <span className="results-icon">🤖</span>
                <p>Results will appear here after classification</p>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Built with TensorFlow · FastAPI · React &nbsp;|&nbsp; CIFAR-10 Dataset
        </p>
      </footer>
    </div>
  );
}
