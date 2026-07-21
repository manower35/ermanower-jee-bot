---
title: ErManower JEE Bot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ErManower JEE Bot — 100% Free Gradio Space Deployment

This repository contains the Gradio web UI and Telegram background daemon for **ErManower JEE Bot** — an AI Socratic engineering entrance exam tutor for **IIT-JEE**, **TG EAPCET**, and **IPE Board** students.

## Environment Secrets Required
Add these under **Settings -> Variables and secrets** in your Hugging Face Space:

| Secret Name | Value |
|-------------|-------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot Token from @BotFather |
| `GROQ_API_KEY` | Your Groq API Key |
| `GOOGLE_API_KEY` | Your Google API Key (optional) |

## Features
- **Telegram Bot Daemon**: Runs 24/7 in a background thread inside the Space.
- **Gradio Web Interface**: Interactive test chat on Hugging Face web page.
