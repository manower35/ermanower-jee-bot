"""
ErManower JEE Bot — Core Application Entry Point
==================================================
Asynchronous Telegram runtime using python-telegram-bot >= 20.0.
Handles text messages and high-resolution photo inputs.
Offloads synchronous CrewAI workflows to a thread pool executor
to keep the Telegram event loop fully non-blocking.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads env vars

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from crew_orchestrator import run_crew
from utils import VisionExtractionResult, parse_image, parse_text_query

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ermanower")

# ---------------------------------------------------------------------------
# Thread pool for blocking CrewAI calls
# ---------------------------------------------------------------------------

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="crewai")

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
MAX_PHOTO_SIZE_MB = 10


# ---------------------------------------------------------------------------
# Helper: run sync function in executor (non-blocking)
# ---------------------------------------------------------------------------

async def _run_sync(func, *args, **kwargs):
    """Run a synchronous callable in the thread pool executor."""
    loop = asyncio.get_event_loop()
    bound = partial(func, *args, **kwargs)
    return await loop.run_in_executor(_EXECUTOR, bound)


# ---------------------------------------------------------------------------
# Helper: build student input string from vision result
# ---------------------------------------------------------------------------

def _format_vision_for_crew(vision: VisionExtractionResult, original_caption: str = "") -> str:
    """Combine vision extraction and optional caption into a crew-ready string."""
    parts: list[str] = []

    if original_caption:
        parts.append(f"[Student Caption]: {original_caption}")

    if vision.question_summary:
        parts.append(f"[Question Summary]: {vision.question_summary}")

    if vision.raw_text:
        parts.append(f"[Extracted Text]: {vision.raw_text}")

    if vision.equations_latex:
        parts.append("[Equations (LaTeX)]:")
        for eq in vision.equations_latex:
            parts.append(f"  {eq}")

    if vision.diagrams:
        parts.append("[Diagrams]:")
        for diag in vision.diagrams:
            parts.append(f"  - {diag.diagram_type}: {diag.description}")

    if vision.detected_subject:
        parts.append(f"[Detected Subject]: {vision.detected_subject}")

    if vision.detected_exam:
        parts.append(f"[Detected Exam]: {vision.detected_exam}")

    return "\n".join(parts) if parts else original_caption or "(empty input)"


# ---------------------------------------------------------------------------
# /start command handler
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Respond to the /start command with a short, clean welcome message."""
    welcome = (
        "🎓 *ErManower JEE Bot*\n"
        "_*Your AI Socratic Engineering Tutor*_\n\n"
        "Welcome! I help you master concepts for:\n"
        "• *IIT-JEE* (Main & Advanced)\n"
        "• *TG EAPCET*\n"
        "• *IPE Board*\n\n"
        "💡 *How to use:*\n"
        "Send your question as text or snap a photo of a problem. "
        "I'll guide you step-by-step with principles, LaTeX formulas, and tactical hints!\n\n"
        "🚀 *What topic would you like to explore today?*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# /help command handler
# ---------------------------------------------------------------------------

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Respond to the /help command."""
    help_text = (
        "🎓 *ErManower JEE Bot — Quick Help*\n\n"
        "• Type any Maths, Physics, or Chemistry question\n"
        "• Or send a photo of a textbook problem\n\n"
        "I'll analyze the concept, pull relevant NCERT formulas, and guide your next step!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Interactive Menu & Quiz Command Handlers
# ---------------------------------------------------------------------------

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display interactive command & subject selection menu."""
    keyboard = [
        [
            InlineKeyboardButton("⚡ Physics", callback_data="subject_physics"),
            InlineKeyboardButton("🧪 Chemistry", callback_data="subject_chemistry"),
            InlineKeyboardButton("📚 Maths", callback_data="subject_maths"),
        ],
        [
            InlineKeyboardButton("🎯 Practice Interactive Quiz", callback_data="cmd_quiz"),
        ],
        [
            InlineKeyboardButton("❓ Quick Help", callback_data="cmd_help"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎓 ErManower JEE & EAPCET Interactive Menu\n\n"
        "Select a subject to explore high-yield topics or tap Practice Interactive Quiz to test your knowledge!",
        reply_markup=reply_markup,
    )


async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate an interactive multiple choice practice question."""
    quiz_text = (
        "🎯 JEE/EAPCET Practice Challenge (Physics - Laws of Motion)\n\n"
        "Question: A 5 kg block moves on a smooth surface under a force F = 20 N at an angle 60° to the horizontal. "
        "What is the horizontal acceleration of the block?\n\n"
        "A) 1.0 m/s²\n"
        "B) 2.0 m/s²\n"
        "C) 4.0 m/s²\n"
        "D) 0.5 m/s²"
    )
    keyboard = [
        [
            InlineKeyboardButton("A) 1.0 m/s²", callback_data="quiz_opt_A"),
            InlineKeyboardButton("B) 2.0 m/s²", callback_data="quiz_opt_B"),
        ],
        [
            InlineKeyboardButton("C) 4.0 m/s²", callback_data="quiz_opt_C"),
            InlineKeyboardButton("D) 0.5 m/s²", callback_data="quiz_opt_D"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await update.callback_query.message.reply_text(quiz_text, reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(quiz_text, reply_markup=reply_markup)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process clicks on inline keyboard buttons."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("subject_"):
        sub = data.split("_")[1].capitalize()
        await query.message.reply_text(
            f"📚 You selected {sub}!\n\n"
            f"Send any query in {sub} (e.g., 'top 5 {sub.lower()} questions' or 'explain key formula') to get Socratic tutoring!"
        )
    elif data.startswith("quiz_opt_"):
        chosen = data.split("_")[-1]
        if chosen == "B":
            feedback = (
                "✅ Correct! Excellent reasoning!\n\n"
                "1. Key Concept: Only the horizontal component F cos θ accelerates the block.\n"
                "2. Governing Formula: a = (F · cos 60°) / m\n"
                "3. Calculation: F cos 60° = 20 · 0.5 = 10 N -> a = 10 / 5 = 2.0 m/s²."
            )
        else:
            feedback = (
                f"❌ Option {chosen} is not correct.\n\n"
                "1. Tactical Hint: Did you resolve F into horizontal component F · cos(60°)?\n"
                "2. Formula: a = (F · cos θ) / m\n"
                "Try computing with F = 20 N, cos 60° = 0.5, m = 5 kg!"
            )
        await query.message.reply_text(feedback)
    elif data == "cmd_quiz":
        await cmd_quiz(update, context)
    elif data == "cmd_help":
        await cmd_help(update, context)


# ---------------------------------------------------------------------------
# Text message handler
# ---------------------------------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a plain-text student query with conversational memory."""
    user_text = update.message.text.strip()
    if not user_text:
        return

    user = update.effective_user
    logger.info("Text query from %s (id=%d): %s", user.first_name, user.id, user_text[:100])

    # Handle short greetings cleanly without full RAG pipeline
    if user_text.lower() in {"hi", "hello", "hey", "hlo", "namaste"}:
        context.user_data.clear()
        await update.message.reply_text(
            "👋 Hello! I'm ErManower, your JEE & EAPCET Socratic Tutor.\n\n"
            "Ask me any question in Maths, Physics, or Chemistry to get started! 📚",
        )
        return

    # Send typing indicator
    await update.message.chat.send_action("typing")

    # Maintain conversational context for follow-up inputs (e.g. "1", "2", "option A", "v = u+at")
    last_turn = context.user_data.get("last_turn", "")
    if last_turn and len(user_text) < 100:
        query_for_engine = f"Previous Conversation Context:\n{last_turn}\n\nStudent Follow-up / Answer:\n{user_text}"
    else:
        query_for_engine = user_text

    # Execute low-latency Socratic engine
    try:
        response = await _run_sync(run_crew, query_for_engine)
    except Exception as exc:
        logger.error("Socratic engine failed: %s", exc, exc_info=True)
        response = (
            "⚠️ I encountered an issue while processing your question. "
            "Please try again or rephrase your query."
        )

    # Save short memory of this turn for continuous discussion
    context.user_data["last_turn"] = f"Student: {user_text}\nTutor: {response[:350]}"

    # Step 4: Send response (split if > 4096 chars)
    await _send_long_message(update, response)


# ---------------------------------------------------------------------------
# Photo message handler
# ---------------------------------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a photo upload from the student."""
    user = update.effective_user
    caption = update.message.caption or ""
    logger.info("Photo received from %s (id=%d), caption=%r", user.first_name, user.id, caption[:80])

    await update.message.chat.send_action("typing")

    # Step 1: Download the highest-resolution photo from Telegram CDN
    photo = update.message.photo[-1]  # Highest resolution variant
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    image_bytes = bytes(image_bytes)

    logger.info("Downloaded photo: file_id=%s, size=%d bytes", photo.file_id, len(image_bytes))

    # Determine MIME type (Telegram compresses to JPEG)
    mime_type = "image/jpeg"

    # Step 2: Parse image with Gemini Vision
    try:
        vision_result = await _run_sync(parse_image, image_bytes, mime_type)
    except Exception as exc:
        logger.error("Vision parsing failed: %s", exc, exc_info=True)
        await update.message.reply_text(
            "⚠️ I couldn't process this image. Please ensure it's a clear photo "
            "of a question and try again."
        )
        return

    logger.info(
        "Vision extraction complete — subject=%s, exam=%s, equations=%d, diagrams=%d",
        vision_result.detected_subject,
        vision_result.detected_exam,
        len(vision_result.equations_latex),
        len(vision_result.diagrams),
    )

    # Step 3: Build crew input from vision result
    crew_input = _format_vision_for_crew(vision_result, original_caption=caption)

    # Step 4: Run CrewAI pipeline in thread pool
    await update.message.chat.send_action("typing")
    try:
        response = await _run_sync(run_crew, crew_input)
    except Exception as exc:
        logger.error("CrewAI pipeline failed: %s", exc, exc_info=True)
        response = (
            "⚠️ I encountered an issue while analyzing your photo. "
            "Please try again or type your question as text."
        )

    # Step 5: Send response
    await _send_long_message(update, response)


# ---------------------------------------------------------------------------
# Helper: send long messages (Telegram 4096-char limit)
# ---------------------------------------------------------------------------

async def _send_long_message(update: Update, text: str) -> None:
    """Split and send a response that may exceed Telegram's message length limit."""
    max_len = 4000  # Leave margin for safety

    if len(text) <= max_len:
        await update.message.reply_text(text)
        return

    # Split on paragraph boundaries where possible
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Try to split at a double newline near the limit
        split_pos = text.rfind("\n\n", 0, max_len)
        if split_pos == -1:
            split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    for i, chunk in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(0.3)  # Rate limit courtesy
        await update.message.reply_text(chunk)


# ---------------------------------------------------------------------------
# Document / sticker / other handler (graceful rejection)
# ---------------------------------------------------------------------------

async def handle_unsupported(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inform the user about unsupported input types."""
    await update.message.reply_text(
        "📌 I currently support *text messages* and *photos* of questions.\n"
        "Please send your question as text or a clear photo!",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and notify the user if possible."""
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ An unexpected error occurred. Please try again shortly."
        )


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and launch the Telegram application."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is not set. Exiting.")
        sys.exit(1)

    logger.info("Starting ErManower JEE Bot...")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    # Register handlers (order matters — first match wins)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(
        filters.Document.ALL | filters.Sticker.ALL | filters.VIDEO | filters.VOICE,
        handle_unsupported,
    ))

    # Global error handler
    app.add_error_handler(error_handler)

    logger.info("ErManower JEE Bot is polling for updates...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
