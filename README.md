
# TheraVox AI — Multimodal Emotion Analysis & Mental Wellness Platform

TheraVox AI is a full-stack multimodal emotion analysis and mental wellness platform that detects emotions through **text**, **audio**, and **facial video** analysis. It features an AI wellness companion chatbot, guided journaling, wellness tracking, real-time crisis detection, gamified self-care tools, and more.

> **Detected Emotions (7):** Happy 😄 · Sad 😢 · Angry 😡 · Fear 😨 · Disgust 🤢 · Surprise 😮 · Neutral 😐

---

## Features

### Multimodal Emotion Detection
- **Text Analysis** — Transformer-based emotion classification (`j-hartmann/emotion-english-distilroberta-base`) with lexicon-based fallback (emoji detection, negation handling, intensifiers).
- **Audio Analysis** — Wav2Vec2 XLS-R model (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) with acoustic rule-based fallback. Supports WAV, FLAC, OGG uploads and in-browser microphone recording.
- **Facial/Vision Analysis** — DeepFace with mediapipe (primary) and opencv (fallback) detector backends. Temporal smoothing over a 3-frame window and CLAHE preprocessing for low-light conditions.

### AI Wellness Companion (MindfulMind)
- Conversational AI powered by **Groq API** (Llama 3.1 8B Instant).
- Persistent chat sessions with full message history.
- AI-generated session summaries with key themes, action items, and mood arc.

### Guided Journaling
- AI-generated journaling prompts based on current mood.
- Journal submission with automatic emotion analysis and AI reflection.

### Wellness Toolkit
- **Mood Check** — Log mood with emoji.
- **Breathing Coach** — Guided breathing exercises with time tracking.
- **Gratitude Box** — Track daily gratitude entries.
- **Daily Affirmations** — Rotating affirmation display.
- **Focus Timer** — Pomodoro-style focus sessions.

### Crisis Detection & Escalation
- Real-time regex-based crisis signal scanner (~40 patterns across suicidal ideation, self-harm, abuse, severe distress).
- 4 severity levels: LOW, MODERATE, HIGH, CRITICAL.
- Automatic async email escalation to admin/therapist for HIGH/CRITICAL alerts.
- Inline crisis detection in text analysis and chat endpoints.
- Crisis resource database (India, USA, International helplines).

### Emotion Postcards
- Shareable postcard generation with curated quotes, color palettes, and patterns based on detected emotion.

### Gamification & Tracking
- 10 unlockable achievements (First Step, Breath of Fresh Air, Deep Breather, Grateful Heart, Journal Start, and more).
- Streak tracking (consecutive days of wellness activity).
- 7-day habit tracking grid.
- CSV export for wellness data.

### Authentication & Security
- JWT access tokens + httpOnly cookie refresh tokens (with rotation).
- OAuth 2.0 (Google & GitHub).
- bcrypt password hashing.
- Rate limiting via slowapi.
- Single-session policy (login revokes existing refresh tokens).
- Secure cookies in production (httpOnly, secure, SameSite=Lax).

### Feedback System
- Submit feedback (bug, suggestion, general, compliment) with 1–5 rating.
- Async email notifications to admin inbox.

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| Framework | FastAPI 0.115.6 + Uvicorn |
| Language | Python 3.11+ |
| Database | PostgreSQL 16 (async via SQLAlchemy 2.0 + asyncpg) |
| Migrations | Alembic (async-compatible) |
| Text Emotion | HuggingFace Transformers (DistilRoBERTa) |
| Audio Emotion | Wav2Vec2 XLS-R (PyTorch) |
| Facial Emotion | DeepFace (mediapipe + opencv) |
| Chat AI | Groq API (Llama 3.1 8B Instant) |
| Auth | JWT + OAuth 2.0 (Google, GitHub) · bcrypt |
| Rate Limiting | slowapi |
| Email | aiosmtplib (async SMTP) |
| Middleware | GZip compression |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 19 + TypeScript 5.9 |
| Build Tool | Vite 7 |
| Routing | React Router DOM 6 |
| Animations | Framer Motion 12 |
| Screenshot/Export | html-to-image |
| State | React Context (Auth) · useReducer (Wellness, localStorage persistence) |

### Deployment
| Component | Technology |
|---|---|
| Containerization | Docker (multi-stage: Node.js build → Python runtime) |
| Platform | Render.com (IaC via `render.yaml`) |
| Services | Web service (Docker) + PostgreSQL 16 |

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 16

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-org/TheraVox-AI.git
cd TheraVox-AI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up environment variables (see Environment Variables section)

# Run database migrations
alembic upgrade head

# Start the server
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server (proxies /api to backend at :8000)
npm run dev
```

### Docker

```bash
docker build -t theravox-ai .
docker run -p 8000:8000 --env-file .env theravox-ai
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | JWT signing secret |
| `JWT_ALGORITHM` | JWT algorithm (default: `HS256`) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access token TTL (default: `60`) |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token TTL (default: `7`) |
| `GROQ_API_KEY` | Groq AI API key for chat + journal |
| `GROQ_MODEL` | Groq model (default: `llama-3.1-8b-instant`) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth credentials |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth credentials |
| `FRONTEND_URL` | Frontend base URL for OAuth redirects |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` | SMTP configuration |
| `NOTIFY_EMAIL` | Admin/therapist inbox for feedback + crisis alerts |
| `ENVIRONMENT` | `development` or `production` |
| `HOST` / `PORT` | Server bind address |

---

## API Reference

### Authentication — `/api/auth`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Create account (rate-limited: 5/min) |
| POST | `/api/auth/login` | Login, get JWT + refresh cookie (rate-limited: 10/min) |
| POST | `/api/auth/refresh` | Rotate refresh token |
| POST | `/api/auth/logout` | Revoke refresh token, clear cookie |
| GET | `/api/auth/me` | Get current user profile |
| PATCH | `/api/auth/me` | Update name/email |
| POST | `/api/auth/me/password` | Change password |
| GET | `/api/auth/me/stats` | Account statistics |

### OAuth — `/api/auth`
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/auth/google/authorize` | Google consent redirect |
| GET | `/api/auth/google/callback` | Google OAuth callback |
| GET | `/api/auth/github/authorize` | GitHub consent redirect |
| GET | `/api/auth/github/callback` | GitHub OAuth callback |

### Emotion Analysis
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze_text` | Text emotion analysis (includes inline crisis detection) |
| POST | `/api/analyze_audio` | Audio emotion analysis (file upload) |
| GET | `/api/audio_status` | Audio analyzer runtime status |
| POST | `/api/analyze_frame` | Facial emotion analysis (base64 image) |

### AI Chat — `/api/chat`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Send message to MindfulMind AI (includes crisis detection) |
| GET | `/api/chat/sessions` | List all chat sessions |
| GET | `/api/chat/sessions/{id}` | Load session message history |
| DELETE | `/api/chat/sessions/{id}` | Delete a session |
| POST | `/api/chat/sessions/{id}/summary` | Generate AI session summary |
| GET | `/api/chat/sessions/{id}/summary` | Retrieve session summary |

### Guided Journal — `/api/journal`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/journal/prompt` | Get AI-generated journaling prompt |
| POST | `/api/journal/submit` | Submit entry → emotion analysis + AI reflection |

### Wellness — `/api/wellness`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/wellness/entries` | Create wellness entry |
| GET | `/api/wellness/entries` | List entries (filterable, paginated) |

### Feedback — `/api/feedback`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/feedback` | Submit feedback |
| GET | `/api/feedback` | List user's feedback |

### Emotion Postcard — `/api/postcard`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/postcard` | Generate shareable postcard for a detected emotion |

### Crisis Detection — `/api/crisis`
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/crisis/scan` | Scan text for crisis signals |
| GET | `/api/crisis/alerts` | Paginated list of user's crisis alerts |
| GET | `/api/crisis/alerts/{id}` | Get alert detail |
| POST | `/api/crisis/alerts/{id}/resolve` | Mark alert as resolved |
| GET | `/api/crisis/resources` | Get crisis helpline resources |

### System
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (model readiness status) |
| POST | `/save_screenshot` | Save screenshot (base64 or file upload) |

---

## Project Structure

```
TheraVox-AI/
├── main.py                    # FastAPI application entry point
├── alembic.ini                # Alembic configuration
├── requirements.txt           # Python dependencies
├── requirements-deploy.txt    # Production dependencies
├── dockerfile                 # Multi-stage Docker build
├── render.yaml                # Render.com deployment config
├── alembic/                   # Database migrations
│   └── versions/              # Migration scripts
├── app/
│   ├── api/                   # API route handlers
│   │   ├── router.py          # Central route aggregator
│   │   ├── auth.py            # Authentication endpoints
│   │   ├── oauth.py           # OAuth 2.0 (Google, GitHub)
│   │   ├── chat.py            # AI chat companion
│   │   ├── text.py            # Text emotion analysis
│   │   ├── audio.py           # Audio emotion analysis
│   │   ├── vision.py          # Facial emotion analysis
│   │   ├── journal.py         # Guided journaling
│   │   ├── wellness.py        # Wellness entries
│   │   ├── feedback.py        # Feedback system
│   │   ├── postcard.py        # Emotion postcards
│   │   ├── crisis.py          # Crisis detection & alerts
│   │   ├── system.py          # Health check & screenshots
│   │   └── dependencies.py    # FastAPI dependency injection
│   ├── auth/                  # JWT & auth utilities
│   ├── core/                  # Config, lifespan, rate limiter
│   ├── db/                    # Database models & connection
│   ├── models/                # Pydantic schemas
│   ├── services/              # ML & business logic services
│   │   ├── text_analyzer.py   # Text emotion (Transformer + lexicon)
│   │   ├── audio_analyzer.py  # Audio emotion (Wav2Vec2 + acoustic)
│   │   ├── vision_analyzer.py # Facial emotion (DeepFace)
│   │   ├── crisis_detector.py # Crisis signal detection
│   │   ├── crisis_escalation.py # Async email escalation
│   │   └── email_service.py   # Async email notifications
│   └── utils/                 # Emotion mapping, file utilities
├── frontend/                  # React + TypeScript SPA
│   ├── src/
│   │   ├── pages/             # 14 pages (Login, Register, Home, Text, Audio, Vision, Chat, Wellness, etc.)
│   │   ├── components/        # Shared, layout, profile, and wellness components
│   │   ├── contexts/          # AuthContext (JWT state management)
│   │   ├── hooks/             # Audio recorder, camera, wellness store
│   │   ├── lib/               # API clients, wellness storage, CSV export, WAV encoder
│   │   └── styles/            # CSS styles
│   └── public/                # Static assets
├── static/                    # Built frontend assets (served by FastAPI)
├── logs/                      # Application logs
└── screenshots/               # Saved screenshots
```

---

## Architecture Highlights

- **Lazy model loading** — All ML models load on first request for fast cold boot.
- **Singleton dependency injection** — Services are lazy-initialized singletons via FastAPI DI.
- **Thread safety** — Audio model loading uses a threading lock; crisis detector is stateless.
- **CPU-optimized** — Thread limits configurable via env vars (`TORCH_NUM_THREADS`, `OMP_NUM_THREADS`, etc.).
- **Executor offloading** — All ML inference runs via `run_in_executor` to avoid blocking the async event loop.
- **Graceful degradation** — Transformer models fall back to rule-based analysis; Groq journal prompts fall back to curated static prompts.
- **Inline crisis detection** — Text analysis and chat endpoints automatically scan and surface crisis signals.

---

## Database Schema

| Table | Purpose |
|---|---|
| `users` | User accounts (email/password + OAuth provider) |
| `refresh_tokens` | SHA-256 hashed refresh tokens with revocation support |
| `wellness_entries` | Wellness logs (journal, mood_log, gratitude, activity) |
| `feedback` | User feedback with type and rating |
| `chat_sessions` | Named conversation sessions |
| `chat_messages` | Individual chat messages (user/assistant) |
| `session_summaries` | AI-generated summaries with themes, actions, mood arc |
| `crisis_alerts` | Crisis risk events with severity and escalation status |

---

## License

MIT — see `LICENSE`.

**Key Libraries:** FastAPI, React, DeepFace, HuggingFace Transformers, PyTorch, Wav2Vec2, Groq, SQLAlchemy, Framer Motion