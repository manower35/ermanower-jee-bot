"""
ErManower JEE Bot — Gradio Web Interface & Telegram Background Host
====================================================================
Runs both:
  1. Telegram Bot (python-telegram-bot) in a background thread 24/7.
  2. Gradio Web Interface for hosting on Render.
  3. Self-ping keep-alive thread to prevent Render free tier sleep.
"""

import asyncio
import os
import threading
import time
import traceback
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
        traceback.print_exc()

# Launch Telegram bot background thread
bot_thread = threading.Thread(target=_start_telegram_bot, daemon=True)
bot_thread.start()

# ---------------------------------------------------------------------------
# Keep-Alive Self-Ping (Prevents Render Free Tier Sleep)
# ---------------------------------------------------------------------------
def _keep_alive():
    """Ping own URL every 10 minutes to prevent Render free tier shutdown."""
    import httpx
    url = os.environ.get("RENDER_EXTERNAL_URL", "https://ermanower-jee-bot.onrender.com")
    print(f"[Keep-Alive] Starting self-ping loop for: {url}")
    while True:
        time.sleep(600)  # Wait 10 minutes
        try:
            resp = httpx.get(url, timeout=30)
            print(f"[Keep-Alive] Ping OK — status={resp.status_code}")
        except Exception as err:
            print(f"[Keep-Alive] Ping failed: {err}")

keep_alive_thread = threading.Thread(target=_keep_alive, daemon=True)
keep_alive_thread.start()

# ---------------------------------------------------------------------------
# Gradio Web Interface
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
    port = int(os.environ.get("PORT", 7860))
    print(f"[Gradio Host] Launching web interface on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
