"""
ErManower JEE Bot — Gradio Web Interface & Telegram Background Host
====================================================================
Runs both:
  1. Telegram Bot (python-telegram-bot) in a background thread 24/7.
  2. Gradio Web Interface for 100% free hosting on Hugging Face Spaces (Gradio SDK).
"""

import asyncio
import os
import threading
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from crew_orchestrator import run_crew
import main as bot_main

# ---------------------------------------------------------------------------
# Background Thread for Telegram Bot Polling
# ---------------------------------------------------------------------------
def _start_telegram_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("[Gradio Host] TELEGRAM_BOT_TOKEN not set in environment secrets.")
        return
    print("[Gradio Host] Starting Telegram Bot polling loop in background...")
    try:
        bot_main.main()
    except Exception as exc:
        print(f"[Gradio Host] Telegram bot error: {exc}")

# Launch Telegram bot background thread
bot_thread = threading.Thread(target=_start_telegram_bot, daemon=True)
bot_thread.start()

# ---------------------------------------------------------------------------
# Gradio Web Interface for Hugging Face Preview
# ---------------------------------------------------------------------------
def respond(message: str, history: list) -> str:
    """Pass web chat message to ErManower Socratic engine."""
    if not message or not message.strip():
        return "Please enter a valid question."
    return run_crew(message.strip())

demo = gr.ChatInterface(
    fn=respond,
    title="🎓 ErManower JEE Bot — AI Socratic Engineering Tutor",
    description=(
        "**Target Syllabus:** IIT-JEE (Main & Advanced), TG EAPCET, and Telangana IPE Board.\n\n"
        "⚡ **Telegram Bot Status:** Running 24/7 in the background!"
    ),
    examples=[
        "What is the discriminant of a quadratic equation?",
        "Explain Markovnikov addition rule in Organic Chemistry",
        "State Newton's second law of motion",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
