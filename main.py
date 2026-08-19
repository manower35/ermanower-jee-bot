"""
ErManower JEE Bot — Core Application Entry Point (Advanced AI Edition)
=======================================================================
Asynchronous Telegram runtime using python-telegram-bot >= 20.0.
Features:
  • 3-Tier Adaptive Difficulty Quizzes (Level 1 EAPCET, Level 2 JEE Main, Level 3 Advanced)
  • Interactive "Show Hint" & Next Question Buttons
  • Dedicated Commands: /formula, /trick, /compare, /mistakes, /stats, /quiz, /menu
  • Real-time Student Accuracy Scorecard & Streak Tracker
  • Multimodal Vision OCR with Google Gemini 2.0 Flash
  • High-Speed Socratic AI Engine with Groq Llama 3.3 70B
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads env vars

import asyncio
import json
import logging
import os
import random
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
from utils import VisionExtractionResult, parse_image, parse_text_query, get_topic_diagram_url, get_topic_diagram_info

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
# Thread pool for blocking AI calls
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
# Advanced 3-Tier Multi-Level QUIZ_BANK
# ---------------------------------------------------------------------------

QUIZ_BANK = [
    # LEVEL 1: TG EAPCET / IPE BOARD (Formula & Concept Application)
    {
        "id": "q_eapcet_1",
        "level": "🟢 Level 1 (TG EAPCET / IPE)",
        "subject": "Physics",
        "title": "Laws of Motion & Friction",
        "diagram_url": "https://quickchart.io/chart?c=%7Btype%3A%27radar%27%2Cdata%3A%7Blabels%3A%5B%27F_x%27%2C%27F_y%27%2C%27Normal%27%2C%27Gravity%27%5D%2Cdatasets%3A%5B%7Blabel%3A%27Forces%27%2Cdata%3A%5B10%2C17.3%2C49%2C49%5D%7D%5D%7D&title=Free+Body+Diagram",
        "question": "A 5 kg block rests on a smooth horizontal surface. A force F = 20 N acts at 60° to the horizontal. What is the horizontal acceleration?",
        "options": {"A": "1.0 m/s²", "B": "2.0 m/s²", "C": "4.0 m/s²", "D": "0.5 m/s²"},
        "correct": "B",
        "hint": "Resolve force along horizontal: F_x = F · cos(60°). Then apply Newton's 2nd Law (F_x = m · a).",
        "explanation": "1. Key Concept: Horizontal component F · cos 60° accelerates the body.\n2. Formula: a = (F · cos 60°) / m\n3. Calculation: F · cos 60° = 20 · 0.5 = 10 N ➔ a = 10 / 5 = 2.0 m/s²."
    },
    {
        "id": "q_eapcet_2",
        "level": "🟢 Level 1 (TG EAPCET / IPE)",
        "subject": "Maths",
        "title": "Vectors & 3D Geometry",
        "diagram_url": "https://quickchart.io/chart?c=%7Btype%3A%27line%27%2Cdata%3A%7Blabels%3A%5B%27X%27%2C%27Y%27%2C%27Z%27%5D%2Cdatasets%3A%5B%7Blabel%3A%27Vector+A%27%2Cdata%3A%5B2%2C1%2C-1%5D%7D%2C%7Blabel%3A%27Vector+B%27%2Cdata%3A%5B1%2C-1%2C1%5D%7D%5D%7D&title=Vector+Dot+Product",
        "question": "If vector a = 2i + j - k and b = i - j + k, what is the scalar dot product a · b?",
        "options": {"A": "0 (Perpendicular)", "B": "2", "C": "-1", "D": "4"},
        "correct": "A",
        "hint": "Calculate ax·bx + ay·by + az·bz. If sum is 0, the angle between vectors is 90°.",
        "explanation": "1. Formula: a · b = (2)(1) + (1)(-1) + (-1)(1) = 2 - 1 - 1 = 0.\n2. Conclusion: Since a · b = 0, vectors are mutually perpendicular."
    },
    {
        "id": "q_eapcet_3",
        "level": "🟢 Level 1 (TG EAPCET / IPE)",
        "subject": "Chemistry",
        "title": "Solutions & Colligative Properties",
        "question": "Which of the following 0.1 M aqueous solutions will exhibit the MAXIMUM boiling point elevation?",
        "options": {"A": "0.1 M Glucose", "B": "0.1 M NaCl", "C": "0.1 M BaCl2", "D": "0.1 M Urea"},
        "correct": "C",
        "hint": "Elevation in boiling point deltaTb = i · Kb · m. Find which solute gives the highest van 't Hoff factor (i).",
        "explanation": "1. Key Concept: deltaTb depends on van 't Hoff factor (i).\n2. Ionization: BaCl2 dissociates into Ba2+ + 2Cl- (i = 3).\n3. Comparison: Glucose (i=1), NaCl (i=2), BaCl2 (i=3). BaCl2 gives highest boiling point."
    },

    # LEVEL 2: JEE MAIN / NEET (Multi-Concept & Numerical Traps)
    {
        "id": "q_jee_1",
        "level": "🟡 Level 2 (JEE Main / NEET)",
        "subject": "Physics",
        "title": "Current Electricity (Kirchhoff's & Resistance)",
        "diagram_url": "https://quickchart.io/chart?c=%7Btype%3A%27bar%27%2Cdata%3A%7Blabels%3A%5B%27R1%27%2C%27R2%27%2C%27R3%27%5D%2Cdatasets%3A%5B%7Blabel%3A%27Resistance+(Ohms)%27%2Cdata%3A%5B2%2C3%2C6%5D%7D%5D%7D&title=Parallel+Resistance+Network",
        "question": "Three resistors of 2 Ω, 3 Ω, and 6 Ω are connected in parallel across a 12 V battery. What is the total current drawn from the battery?",
        "options": {"A": "6 A", "B": "12 A", "C": "4 A", "D": "2 A"},
        "correct": "B",
        "hint": "First calculate equivalent parallel resistance: 1/Req = 1/2 + 1/3 + 1/6. Then apply Ohm's law I = V / Req.",
        "explanation": "1. Parallel Req: 1/Req = 3/6 + 2/6 + 1/6 = 6/6 = 1 ➔ Req = 1 Ω.\n2. Total Current: I = V / Req = 12 V / 1 Ω = 12 A."
    },
    {
        "id": "q_jee_2",
        "level": "🟡 Level 2 (JEE Main / NEET)",
        "subject": "Chemistry",
        "title": "Coordination Chemistry (Crystal Field Theory)",
        "diagram_url": "https://quickchart.io/chart?c=%7Btype%3A%27bar%27%2Cdata%3A%7Blabels%3A%5B%27d1%27%2C%27d2%27%2C%27d3%27%2C%27d4%27%2C%27d5%27%2C%27d6%27%5D%2Cdatasets%3A%5B%7Blabel%3A%273d+Spin%27%2Cdata%3A%5B1%2C1%2C1%2C1%2C1%2C1%5D%7D%5D%7D&title=Octahedral+Spin+State",
        "question": "What is the spin-only magnetic moment (in Bohr Magnetons) of [Fe(H2O)6]2+ ion? (Fe atomic number Z = 26)",
        "options": {"A": "2.83 BM", "B": "4.90 BM", "C": "5.92 BM", "D": "0.00 BM"},
        "correct": "B",
        "hint": "Fe(II) has 3d6 configuration. H2O is a weak field ligand (high spin), so pairing does not occur in t2g. Count unpaired electrons (n).",
        "explanation": "1. Electron configuration: Fe2+ = 3d6 (t2g4 eg2 in weak field) ➔ n = 4 unpaired electrons.\n2. Formula: μ = √(n(n + 2)) = √(4 · 6) = √24 ≈ 4.90 BM."
    },
    {
        "id": "q_jee_3",
        "level": "🟡 Level 2 (JEE Main / NEET)",
        "subject": "Maths",
        "title": "Calculus (Definite Integration)",
        "question": "Evaluate integral I = ∫[0 to π/2] (sin x) / (sin x + cos x) dx:",
        "options": {"A": "π", "B": "π / 2", "C": "π / 4", "D": "0"},
        "correct": "C",
        "hint": "Apply Kings Property: ∫[0 to a] f(x) dx = ∫[0 to a] f(a - x) dx. Then add 2I = ∫[0 to π/2] 1 dx.",
        "explanation": "1. King's Rule: I = ∫ (cos x) / (cos x + sin x) dx.\n2. Adding: 2I = ∫[0 to π/2] 1 dx = [x][0 to π/2] = π/2.\n3. Result: I = π / 4."
    },

    # LEVEL 3: JEE ADVANCED (Calculus Dynamics & Multi-Step Derivations)
    {
        "id": "q_adv_1",
        "level": "🔴 Level 3 (JEE Advanced)",
        "subject": "Physics",
        "title": "Rotational Mechanics (Rolling without Slipping)",
        "question": "A solid cylinder and a hollow sphere of equal mass and radius roll down an inclined plane from rest without slipping. Which reaches the bottom first?",
        "options": {"A": "Solid Cylinder", "B": "Hollow Sphere", "C": "Both together", "D": "Depends on incline angle"},
        "correct": "A",
        "hint": "Acceleration down incline is a = (g · sin θ) / (1 + I / (m·R²)). The body with smaller moment of inertia ratio (I / mR²) has higher acceleration.",
        "explanation": "1. Moment of Inertia: Solid Cylinder = 1/2 mR² (k=0.5). Hollow Sphere = 2/3 mR² (k=0.67).\n2. Acceleration: a = (g sin θ) / (1 + k). Smaller k yields larger acceleration.\n3. Conclusion: Solid cylinder has higher acceleration and reaches the bottom first."
    },
    {
        "id": "q_adv_2",
        "level": "🔴 Level 3 (JEE Advanced)",
        "subject": "Chemistry",
        "title": "Organic Chemistry (Aromaticity & Carbocation Stability)",
        "question": "Which of the following carbocations is non-aromatic / anti-aromatic and least stable?",
        "options": {"A": "Tropylium cation (C7H7+)", "B": "Cyclopropenyl cation (C3H3+)", "C": "Cyclopentadienyl cation (C5H5+)", "D": "Benzyl cation (C7H7+)"},
        "correct": "C",
        "hint": "Check Huckel's rule: (4n + 2) pi electrons = Aromatic, (4n) pi electrons = Anti-Aromatic (highly unstable).",
        "explanation": "1. Cyclopentadienyl cation has 4 pi electrons (n=1 in 4n) in a planar conjugated ring.\n2. By Huckel's rule, 4 pi electrons = Anti-Aromatic, making it extremely unstable."
    }
]


# ---------------------------------------------------------------------------
# /start & /menu command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Respond to /start with full interactive workflow guide."""
    welcome = (
        "🎓 *ErManower JEE & EAPCET Bot (Advanced AI Edition)*\n"
        "_*Your 24/7 AI Socratic Engineering & Medical Entrance Tutor*_\n\n"
        "Welcome! I help you master **IIT-JEE (Main & Adv)**, **TG EAPCET**, and **NEET**.\n\n"
        "⚡ *Quick AI Power Commands:*\n"
        "• `/formula <topic>` ➔ Instant Compact Formula Card\n"
        "• `/trick <topic>` ➔ 10-Second Exam Shortcuts & Elimination\n"
        "• `/compare <A vs B>` ➔ Comparative Tabular Breakdown\n"
        "• `/mistakes <topic>` ➔ Top 5 Negative Marking Traps\n"
        "• `/quiz` ➔ 3-Tier Adaptive PYQ Challenge with Hints\n"
        "• `/stats` ➔ View your Accuracy Scorecard & Streak\n"
        "• `/menu` ➔ Open Interactive Subject Hub\n\n"
        "📸 *Photo OCR:* Snap any question from your book for instant Socratic guidance!\n\n"
        "🚀 *Choose an action below or ask any question to get started:*"
    )
    keyboard = [
        [
            InlineKeyboardButton("🎛️ Subject Menu", callback_data="cmd_menu"),
            InlineKeyboardButton("🎯 3-Tier PYQ Quiz", callback_data="cmd_quiz"),
        ],
        [
            InlineKeyboardButton("🧮 Formula Cards", callback_data="cmd_formulas_hub"),
            InlineKeyboardButton("⚡ Speed Tricks", callback_data="cmd_tricks_hub"),
        ],
        [
            InlineKeyboardButton("📊 My Scorecard", callback_data="cmd_stats"),
            InlineKeyboardButton("❓ Quick Help", callback_data="cmd_help"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome, reply_markup=reply_markup, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Respond to /help command."""
    help_text = (
        "🎓 *ErManower JEE Bot — Command Reference*\n\n"
        "• `/quiz` ➔ Practice 10-Year High-Yield MCQs (Level 1, 2, 3)\n"
        "• `/formula <topic>` ➔ Instant formula cheat sheet\n"
        "• `/trick <topic>` ➔ Speed elimination shortcuts\n"
        "• `/compare <topic1 vs topic2>` ➔ Concept comparison\n"
        "• `/mistakes <topic>` ➔ Avoid negative marking traps\n"
        "• `/stats` ➔ Your personal accuracy & streak\n"
        "• `/menu` ➔ Interactive subject dashboard\n\n"
        "💡 *Tip:* You can type questions naturally or send photos anytime!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display interactive command & subject selection menu."""
    keyboard = [
        [
            InlineKeyboardButton("⚡ Physics Hub", callback_data="subject_physics"),
            InlineKeyboardButton("🧪 Chemistry Hub", callback_data="subject_chemistry"),
        ],
        [
            InlineKeyboardButton("📚 Maths Hub", callback_data="subject_maths"),
            InlineKeyboardButton("🧬 Biology Hub", callback_data="subject_biology"),
        ],
        [
            InlineKeyboardButton("🎯 Start 3-Tier PYQ Quiz", callback_data="cmd_quiz"),
            InlineKeyboardButton("📊 Accuracy Scorecard", callback_data="cmd_stats"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = (
        "🎓 *ErManower Interactive Hub*\n\n"
        "Select a subject hub below to explore high-yield formulas, tricks, and 10-year PYQ trends!"
    )
    if update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    elif update.message:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Dedicated Power AI Commands (/formula, /trick, /compare, /mistakes, /stats)
# ---------------------------------------------------------------------------

async def cmd_formula(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate formula card for specified topic."""
    topic = " ".join(context.args) if context.args else "Thermodynamics and Optics"
    await update.message.chat.send_action("typing")
    query = f"Generate a compact formula card for: {topic}"
    response = await _run_sync(run_crew, query)
    await _send_long_message(update, response)


async def cmd_trick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate 10-second speed shortcut for specified topic."""
    topic = " ".join(context.args) if context.args else "Projectile Motion and Collisions"
    await update.message.chat.send_action("typing")
    query = f"Teach 10-second speed tricks and option elimination for: {topic}"
    response = await _run_sync(run_crew, query)
    await _send_long_message(update, response)


async def cmd_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate comparative breakdown between two concepts."""
    topic = " ".join(context.args) if context.args else "Isothermal vs Adiabatic processes"
    await update.message.chat.send_action("typing")
    query = f"Compare and distinguish: {topic}"
    response = await _run_sync(run_crew, query)
    await _send_long_message(update, response)


async def cmd_mistakes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate top negative marking traps for specified topic."""
    topic = " ".join(context.args) if context.args else "Ray Optics and Electrostatics"
    await update.message.chat.send_action("typing")
    query = f"What are the top negative marking mistakes and traps in: {topic}"
    response = await _run_sync(run_crew, query)
    await _send_long_message(update, response)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display student's quiz scorecard, streak, and target rank estimation."""
    stats = context.user_data.get("stats", {"attempted": 0, "correct": 0, "streak": 0, "best_streak": 0})
    attempted = stats["attempted"]
    correct = stats["correct"]
    streak = stats["streak"]
    best = stats["best_streak"]
    accuracy = (correct / attempted * 100) if attempted > 0 else 0

    if accuracy >= 80 and attempted >= 3:
        rank_est = "🔥 Rank < 5,000 in JEE Main / Top 500 in TG EAPCET!"
    elif accuracy >= 60 and attempted >= 3:
        rank_est = "⚡ Rank ~ 15,000 - 25,000 in JEE Main (Strong Foundation)"
    else:
        rank_est = "🌱 Building Fundamentals — Keep practicing daily PYQs!"

    scorecard = (
        "📊 *Your ErManower Performance Scorecard*\n\n"
        f"• **Total PYQs Attempted:** `{attempted}`\n"
        f"• **Correct Solutions:** `{correct}`\n"
        f"• **Accuracy Rate:** `{accuracy:.1f}%`\n"
        f"• **Current Streak:** `{streak} in a row 🔥`\n"
        f"• **Best Streak:** `{best} in a row 🏆`\n\n"
        f"🎯 *Estimated Trajectory:*\n_{rank_est}_\n\n"
        "Tap `/quiz` to practice another 10-year challenge!"
    )
    if update.callback_query:
        await update.callback_query.message.reply_text(scorecard, parse_mode="Markdown")
    elif update.message:
        await update.message.reply_text(scorecard, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Interactive 3-Tier Quiz Engine Handler
# ---------------------------------------------------------------------------

async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a 3-tier adaptive PYQ practice challenge with Hint and Diagram."""
    quiz_item = random.choice(QUIZ_BANK)
    q_id = quiz_item["id"]

    # Send visual diagram photo if available
    if "diagram_url" in quiz_item:
        try:
            target = update.callback_query.message if update.callback_query else update.message
            await target.reply_photo(
                photo=quiz_item["diagram_url"],
                caption=f"🖼️ *Visual Diagram | {quiz_item['title']}*",
                parse_mode="Markdown"
            )
        except Exception as err:
            logger.warning("Quiz photo send error: %s", err)

    quiz_text = (
        f"🎯 *{quiz_item['level']}*\n"
        f"📚 *{quiz_item['title']}*\n\n"
        f"**Question:** {quiz_item['question']}\n\n"
        f"A) {quiz_item['options']['A']}\n"
        f"B) {quiz_item['options']['B']}\n"
        f"C) {quiz_item['options']['C']}\n"
        f"D) {quiz_item['options']['D']}"
    )

    keyboard = [
        [
            InlineKeyboardButton(f"A) {quiz_item['options']['A']}", callback_data=f"qans_{q_id}_A"),
            InlineKeyboardButton(f"B) {quiz_item['options']['B']}", callback_data=f"qans_{q_id}_B"),
        ],
        [
            InlineKeyboardButton(f"C) {quiz_item['options']['C']}", callback_data=f"qans_{q_id}_C"),
            InlineKeyboardButton(f"D) {quiz_item['options']['D']}", callback_data=f"qans_{q_id}_D"),
        ],
        [
            InlineKeyboardButton("💡 Show Hint", callback_data=f"qhint_{q_id}"),
            InlineKeyboardButton("🔄 Next Question", callback_data="cmd_quiz"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        await update.callback_query.message.reply_text(quiz_text, reply_markup=reply_markup, parse_mode="Markdown")
    elif update.message:
        await update.message.reply_text(quiz_text, reply_markup=reply_markup, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Callback Query Handler (Button Clicks)
# ---------------------------------------------------------------------------

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process clicks on inline keyboard buttons."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("subject_"):
        sub = data.split("_")[1].capitalize()
        await query.message.reply_text(
            f"📚 *{sub} Hub Active!*\n\n"
            f"Try asking:\n"
            f"• `/formula {sub}` ➔ Formula card\n"
            f"• `/trick {sub}` ➔ Speed solving tricks\n"
            f"• `top 5 {sub.lower()} 10-year pyqs` ➔ High-yield questions\n"
            f"• `/quiz` ➔ Test your knowledge!",
            parse_mode="Markdown"
        )
    elif data.startswith("qhint_"):
        q_id = data.split("_")[1]
        item = next((q for q in QUIZ_BANK if q["id"] == q_id), QUIZ_BANK[0])
        await query.message.reply_text(
            f"💡 *Tactical Hint for {item['title']}:*\n\n_{item['hint']}_\n\n"
            "Now pick your answer above! 🎯",
            parse_mode="Markdown"
        )
    elif data.startswith("qans_"):
        parts = data.split("_")
        q_id = parts[1]
        chosen = parts[2]

        item = next((q for q in QUIZ_BANK if q["id"] == q_id), QUIZ_BANK[0])
        stats = context.user_data.setdefault("stats", {"attempted": 0, "correct": 0, "streak": 0, "best_streak": 0})
        stats["attempted"] += 1

        if chosen == item["correct"]:
            stats["correct"] += 1
            stats["streak"] += 1
            if stats["streak"] > stats["best_streak"]:
                stats["best_streak"] = stats["streak"]
            feedback = (
                f"✅ *Correct! Outstanding 10-Year PYQ reasoning!* 🔥\n\n"
                f"{item['explanation']}\n\n"
                f"🏆 *Current Streak:* `{stats['streak']} in a row`"
            )
        else:
            stats["streak"] = 0
            feedback = (
                f"❌ *Option {chosen} is incorrect.* (Correct: **{item['correct']}**)\n\n"
                f"💡 *Socratic Derivation & Analysis:*\n\n"
                f"{item['explanation']}"
            )

        next_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🎯 Next Practice Challenge", callback_data="cmd_quiz")]])
        await query.message.reply_text(feedback, reply_markup=next_kb, parse_mode="Markdown")

    elif data == "cmd_quiz":
        await cmd_quiz(update, context)
    elif data == "cmd_menu":
        await cmd_menu(update, context)
    elif data == "cmd_stats":
        await cmd_stats(update, context)
    elif data == "cmd_help":
        await cmd_help(update, context)
    elif data == "cmd_formulas_hub":
        await query.message.reply_text("🧮 Send `/formula <topic>` (e.g. `/formula optics` or `/formula calculus`) to get an instant formula card!")
    elif data == "cmd_tricks_hub":
        await query.message.reply_text("⚡ Send `/trick <topic>` (e.g. `/trick kinematics` or `/trick organic`) to learn 10-second exam shortcuts!")


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

    # Handle short greetings cleanly
    if user_text.lower() in {"hi", "hello", "hey", "hlo", "namaste"}:
        context.user_data.pop("last_turn", None)
        await update.message.reply_text(
            "👋 Hello! I'm ErManower, your AI JEE, EAPCET & NEET Socratic Tutor.\n\n"
            "Ask any question, send `/formula <topic>`, `/quiz`, or upload a photo to start! 📚",
        )
        return

    # Send typing indicator
    await update.message.chat.send_action("typing")

    # Send visual diagram photo with rich topic caption if query relates to visual concepts
    diag_info = get_topic_diagram_info(user_text)
    if diag_info:
        photo_url, photo_caption = diag_info
        try:
            await update.message.reply_photo(
                photo=photo_url,
                caption=photo_caption,
                parse_mode="Markdown"
            )
        except Exception as err:
            logger.warning("Diagram photo send error: %s", err)

    # Maintain conversational context for follow-ups
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

    # Save short memory of this turn
    context.user_data["last_turn"] = f"Student: {user_text}\nTutor: {response[:350]}"

    # Send response (split if > 4000 chars)
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

    # Download highest-res photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    image_bytes = bytes(image_bytes)

    logger.info("Downloaded photo: file_id=%s, size=%d bytes", photo.file_id, len(image_bytes))
    mime_type = "image/jpeg"

    # Parse image with Gemini Vision
    try:
        vision_result = await _run_sync(parse_image, image_bytes, mime_type)
    except Exception as exc:
        logger.error("Vision parsing failed: %s", exc, exc_info=True)
        await update.message.reply_text(
            "⚠️ I couldn't process this image. Please ensure it's a clear photo "
            "of a question and try again."
        )
        return

    # Build crew input from vision result
    crew_input = _format_vision_for_crew(vision_result, original_caption=caption)

    # Run Socratic AI in thread pool
    await update.message.chat.send_action("typing")
    try:
        response = await _run_sync(run_crew, crew_input)
    except Exception as exc:
        logger.error("CrewAI pipeline failed: %s", exc, exc_info=True)
        response = (
            "⚠️ I encountered an issue while analyzing your photo. "
            "Please try again or type your question as text."
        )

    await _send_long_message(update, response)


# ---------------------------------------------------------------------------
# Helper: send long messages
# ---------------------------------------------------------------------------

async def _send_long_message(update: Update, text: str) -> None:
    """Split and send a response that may exceed Telegram's message length limit."""
    max_len = 4000

    if len(text) <= max_len:
        await update.message.reply_text(text)
        return

    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        split_pos = text.rfind("\n\n", 0, max_len)
        if split_pos == -1:
            split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    for i, chunk in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(0.3)
        await update.message.reply_text(chunk)


# ---------------------------------------------------------------------------
# Document / sticker / other handler
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
    token = os.environ.get("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is not set. Exiting.")
        return

    # Ensure a dedicated asyncio event loop exists in this thread
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as exc:
        logger.warning("Event loop setup in thread: %s", exc)

    logger.info("Starting ErManower JEE Bot (Advanced AI Edition)...")

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .pool_timeout(30.0)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("formula", cmd_formula))
    app.add_handler(CommandHandler("trick", cmd_trick))
    app.add_handler(CommandHandler("compare", cmd_compare))
    app.add_handler(CommandHandler("mistakes", cmd_mistakes))
    app.add_handler(CommandHandler("stats", cmd_stats))

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(
        filters.Document.ALL | filters.Sticker.ALL | filters.VIDEO | filters.VOICE,
        handle_unsupported,
    ))

    app.add_error_handler(error_handler)

    logger.info("ErManower JEE Bot is polling for updates...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        stop_signals=None,  # CRITICAL: Disable signal handling in background thread!
    )


if __name__ == "__main__":
    main()
