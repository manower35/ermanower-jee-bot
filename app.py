"""
ErManower JEE Bot — Gradio Web Interface & Telegram Background Host
====================================================================
Runs both:
  1. Telegram Bot (python-telegram-bot) in a background thread 24/7.
  2. Gradio Web Interface for hosting on Render.
  3. Self-ping keep-alive thread to prevent Render free tier sleep.
  4. Auto-restart if Telegram bot thread crashes.
"""

import asyncio
import os
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Background Thread for Telegram Bot Polling (with auto-restart)
# ---------------------------------------------------------------------------
def _start_telegram_bot():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("[Bot Thread] TELEGRAM_BOT_TOKEN not set. Skipping bot.", flush=True)
        return

    # Delay to let Gradio web server start first
    time.sleep(5)

    while True:
        try:
            print("[Bot Thread] Starting Telegram Bot polling...", flush=True)
            # Import here to avoid circular imports and catch import errors
            import main as bot_main
            bot_main.main()
        except Exception as exc:
            print(f"[Bot Thread] Telegram bot crashed: {exc}", flush=True)
            traceback.print_exc()
            print("[Bot Thread] Restarting in 10 seconds...", flush=True)
            time.sleep(10)  # Wait before auto-restart

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
    print(f"[Keep-Alive] Self-ping target: {url}", flush=True)
    time.sleep(60)  # Wait 1 min for server to fully start
    while True:
        try:
            resp = httpx.get(url, timeout=30)
            print(f"[Keep-Alive] Ping OK — status={resp.status_code}", flush=True)
        except Exception as err:
            print(f"[Keep-Alive] Ping failed: {err}", flush=True)
        time.sleep(600)  # Every 10 minutes

keep_alive_thread = threading.Thread(target=_keep_alive, daemon=True)
keep_alive_thread.start()

# ---------------------------------------------------------------------------
# Gradio Web Interface
# ---------------------------------------------------------------------------
import gradio as gr
from crew_orchestrator import run_crew

def respond(message: str, history: list) -> str:
    """Pass web chat message to ErManower Socratic engine."""
    if not message or not message.strip():
        return "Please enter a valid question."
    try:
        return run_crew(message.strip())
    except Exception as e:
        return f"Error: {e}"

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
    print(f"[Gradio Host] Launching web interface on port {port}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port)
