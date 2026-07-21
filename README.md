---
title: ErManower JEE Bot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ErManower JEE Bot — 24/7 Docker Deployment on Hugging Face Spaces

This repository contains the production Docker build for **ErManower JEE Bot** — an asynchronous, multimodal, Socratic engineering entrance exam tutor for **IIT-JEE**, **TG EAPCET**, and **IPE Board** students.

## Environment Secrets Required
Add these under **Settings -> Repository Secrets** in your Hugging Face Space:

| Secret Name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot Token from @BotFather |
| `GROQ_API_KEY` | Your Groq API Key |
| `GOOGLE_API_KEY` | Your Google API Key (optional) |

## Project Architecture
- `main.py` — Async Telegram runtime (`python-telegram-bot` >= 20.0)
- `utils.py` — Multimodal query parser
- `database.py` — Pre-filtered NCERT, IIT-JEE, TG EAPCET, and IPE vector knowledge store
- `crew_orchestrator.py` — Groq-powered Socratic tutoring engine
- `Dockerfile` — Multi-stage `python:3.11-slim` container
