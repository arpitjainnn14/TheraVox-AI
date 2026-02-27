"""
AI Wellness Companion chat endpoint with persistent session history.

Routes
------
POST   /api/chat                        — send a message (creates or continues a session)
GET    /api/chat/sessions               — list the user's past sessions
GET    /api/chat/sessions/{session_id}  — load all messages for a session
DELETE /api/chat/sessions/{session_id}  — delete a session
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.core.config import get_settings
from app.db.models import ChatMessageDB, ChatSession, User
from app.models.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are MindfulMind, a compassionate AI wellness companion built into TheraVox AI — \
a multimodal emotion analysis and mental wellness platform.

Your role:
- Provide warm, empathetic, non-judgmental emotional support.
- Offer practical wellness techniques: breathing exercises, mindfulness, journaling prompts.
- Help users understand their emotions and develop healthier patterns.
- Suggest relevant wellness tools when appropriate (breathing coach, mood check, gratitude journaling).

Guidelines:
- Be concise: 2–3 short paragraphs per response at most.
- Use a calm, warm, conversational tone — not clinical or overly formal.
- Never diagnose mental health conditions.
- For serious or ongoing mental health concerns, gently recommend speaking with a professional.
- If a user expresses thoughts of self-harm or suicide, immediately provide crisis resources:
    India: AASRA — 9152987821  |  Tele Manas — 1800-891-4416
    International: https://findahelpline.com
- Avoid generic platitudes ("Everything will be fine!"). Be specific and practical.
- You may occasionally reference the user's wellness data when it helps personalise advice.
"""


def _build_system_prompt(context: dict | None) -> str:
    if not context:
        return _BASE_SYSTEM_PROMPT

    lines: list[str] = []
    if context.get("recent_mood"):
        lines.append(f"- The user's most recent logged mood is: {context['recent_mood']}.")
    if context.get("streak") is not None:
        lines.append(f"- Their current wellness streak is {context['streak']} day(s).")
    if context.get("breathing_minutes") is not None:
        lines.append(f"- They have logged {context['breathing_minutes']} total breathing minutes.")

    if not lines:
        return _BASE_SYSTEM_PROMPT

    context_block = "\n".join(lines)
    return f"{_BASE_SYSTEM_PROMPT}\nUser wellness context (for personalisation):\n{context_block}"


def _make_title(first_user_message: str) -> str:
    """Derive a session title from the user's first message."""
    title = first_user_message.strip().replace("\n", " ")
    return title[:97] + "…" if len(title) > 100 else title


# ---------------------------------------------------------------------------
# POST /api/chat — send a message, persist to DB, return reply + session_id
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message to the AI wellness companion",
)
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    settings = get_settings()
    api_key: str = settings.get("groq_api_key", "")
    model: str = settings.get("groq_model", "llama-3.1-8b-instant")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI companion is not configured. Please add GROQ_API_KEY to your .env file.",
        )

    try:
        from groq import Groq
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="groq package is not installed. Run: pip install groq",
        )

    # ------------------------------------------------------------------
    # Resolve or create the session
    # ------------------------------------------------------------------
    session: ChatSession | None = None

    if body.session_id:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == body.session_id,
                ChatSession.user_id == current_user.id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found.",
            )

    # ------------------------------------------------------------------
    # Call Groq
    # ------------------------------------------------------------------
    context_dict = body.context.model_dump() if body.context else None
    system_prompt = _build_system_prompt(context_dict)

    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in body.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid message role: {msg.role!r}. Must be 'user' or 'assistant'.",
            )
        groq_messages.append({"role": msg.role, "content": msg.content})

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=groq_messages,
            max_tokens=512,
            temperature=0.75,
        )
        reply = completion.choices[0].message.content or ""
        reply = reply.strip()
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI companion is temporarily unavailable. Please try again shortly.",
        ) from exc

    # ------------------------------------------------------------------
    # Persist: create session on first message, then save user+assistant msgs
    # ------------------------------------------------------------------
    # The last message in body.messages is always the new user turn
    new_user_content = body.messages[-1].content

    if session is None:
        session = ChatSession(
            user_id=current_user.id,
            title=_make_title(new_user_content),
            message_count=0,
            updated_at=datetime.now(timezone.utc),
        )
        db.add(session)
        await db.flush()  # populate session.id

    # Save new user message + assistant reply
    db.add(ChatMessageDB(session_id=session.id, role="user", content=new_user_content))
    db.add(ChatMessageDB(session_id=session.id, role="assistant", content=reply))
    session.message_count = (session.message_count or 0) + 2
    session.updated_at = datetime.now(timezone.utc)
    # db commit handled by get_db dependency

    return ChatResponse(reply=reply, model=model, session_id=session.id)


# ---------------------------------------------------------------------------
# GET /api/chat/sessions — list user's sessions (most recent first)
# ---------------------------------------------------------------------------

@router.get(
    "/sessions",
    response_model=list[ChatSessionResponse],
    summary="List all past chat sessions for the current user",
)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ChatSessionResponse]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()

    # For each session fetch the last assistant message as a preview
    out: list[ChatSessionResponse] = []
    for s in sessions:
        preview_result = await db.execute(
            select(ChatMessageDB.content)
            .where(
                ChatMessageDB.session_id == s.id,
                ChatMessageDB.role == "assistant",
            )
            .order_by(ChatMessageDB.created_at.desc())
            .limit(1)
        )
        preview_row = preview_result.scalar_one_or_none()
        preview = preview_row[:120] + "…" if preview_row and len(preview_row) > 120 else preview_row

        out.append(
            ChatSessionResponse(
                id=s.id,
                title=s.title,
                message_count=s.message_count,
                created_at=s.created_at,
                updated_at=s.updated_at,
                preview=preview,
            )
        )
    return out


# ---------------------------------------------------------------------------
# GET /api/chat/sessions/{session_id} — load full message history
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetail,
    summary="Load all messages for a specific chat session",
)
async def get_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatSessionDetail:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    msgs_result = await db.execute(
        select(ChatMessageDB)
        .where(ChatMessageDB.session_id == session_id)
        .order_by(ChatMessageDB.created_at)
    )
    messages = msgs_result.scalars().all()

    return ChatSessionDetail(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        messages=[
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/{session_id} — delete a session + all its messages
# ---------------------------------------------------------------------------

@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session and all its messages",
)
async def delete_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    await db.delete(session)
    # cascade deletes all ChatMessageDB rows via FK constraint
