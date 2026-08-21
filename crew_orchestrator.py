"""
ErManower JEE Bot — Advanced Socratic Tutor Engine
===================================================
Ultra-low-latency Socratic Tutor using Groq's 120B/27B models with local
NCERT/JEE RAG context retrieval, human-readable mobile-optimized formatting,
and complete non-truncated outputs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

from database import search_knowledge_bank

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public Orchestration API
# ═══════════════════════════════════════════════════════════════════════════

def run_crew(student_input: str) -> str:
    """
    Execute the Socratic tutor engine with low latency.
    """
    return run_fast_tutor(student_input)


def run_fast_tutor(student_input: str) -> str:
    """
    Ultra-low-latency Advanced Socratic Tutor engine (~0.8s response time).
    Combines local NCERT/JEE RAG retrieval with Groq's high-speed inference
    and human-friendly mobile formatting.
    """
    logger.info("Executing Fast Socratic Tutor for input: %s", student_input[:100])

    # 1. Instant local RAG search (< 5ms)
    rag_results = search_knowledge_bank(query=student_input, top_k=3)
    context_str = ""
    if rag_results:
        valid_results = [item for item in rag_results if item['score'] > 0.5]
        if valid_results:
            context_str = "\n\n--- RELEVANT NCERT / JEE / STATE BOARD REFERENCE ---\n" + "\n\n".join(
                f"• {item['content']}" for item in valid_results
            )

    # 2. Single-pass Groq completion for instant response
    from groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=groq_api_key)

    text_lower = student_input.lower()
    text_clean = student_input.strip().lower()

    # Dynamic Intent Classification
    is_formula_card = bool(re.search(r'\b(formula|formulas|cheat\s*sheet|formula\s*card|key\s*equations?)\b', text_lower))
    is_trick_request = bool(re.search(r'\b(trick|tricks|shortcut|shortcuts|solve\s*fast|speed\s*trick|option\s*elimination|fast\s*method)\b', text_lower))
    is_compare_request = bool(re.search(r'\b(compare|difference\s*between|vs\b|versus|distinguish\s*between)\b', text_lower))
    is_mistakes_request = bool(re.search(r'\b(mistake|mistakes|common\s*errors?|traps?|pitfalls?|negative\s*marking|blunders?)\b', text_lower))
    is_mindmap_request = bool(re.search(r'\b(mind\s*map|mindmap|diagram|flowchart|chart|graph|draw|sketch|map)\b', text_lower))
    is_list_request = bool(re.search(r'\b(top\s*\d+|list|\d+\s*questions|\d+\s*pyqs|important\s*questions|practice\s*questions|question\s*bank|top\s*neet|top\s*jee)\b', text_lower))
    is_proof_request = bool(re.search(r'\b(proof|prove|derive|derivation|step\s*by\s*step\s*derivation|proof\s*of)\b', text_lower))
    is_biology_request = bool(re.search(r'\b(botany|zoology|biology|neet\s*bio|cell|genetics|photosynthesis|plant\s*kingdom|animal\s*kingdom|human\s*physiology|biomolecules|ecology|morphology)\b', text_lower))
    is_short_response = text_clean in [
        "1", "2", "3", "4", "5", "a", "b", "c", "d",
        "option 1", "option 2", "option 3", "option 4",
        "option a", "option b", "option c", "option d"
    ]

    max_tokens = 2048

    if is_formula_card:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — elite IIT-JEE, TG EAPCET, and NEET formula architect.\n"
            "The student wants an instant, beautifully organized FORMULA CHEAT SHEET.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. Format with clean section headers and emojis (never use wide ascii boxes that overflow mobile screens):\n"
            "   📌 CORE EQUATIONS: Write each governing equation on a clear line with plain text symbols.\n"
            "   📐 VARIABLES & SI UNITS: Clearly state symbol meanings and SI units.\n"
            "   ⚡ LIMITS & CONDITIONS: State when equations apply (e.g. non-relativistic, elastic, ideal gas).\n"
            "   🎯 10-YEAR EXAM TIP: 1 tactical substitution or weightage trend for JEE/NEET.\n"
            "   💡 ACTIVE CHALLENGE: 1 quick calculation question testing their memory.\n"
            "2. Always complete every equation and section fully. Never terminate mid-sentence.\n"
            "3. NO ASTERISKS (*) or DOLLAR SIGNS ($). Keep all text clean and mobile-friendly."
        )
        context_str = ""

    elif is_trick_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — Master of Competitive Exam Speed Tactics for IIT-JEE, TG EAPCET & NEET.\n"
            "The student wants 10-SECOND SPEED TRICKS and OPTION ELIMINATION TECHNIQUES specifically for their topic.\n\n"
            "MANDATORY FORMATTING & COMPLETION RULES:\n"
            "1. Structure your answer in 4 concise, high-impact sections:\n"
            "   ⚡ 10-SECOND SHORTCUT: The core formula, ratio trick, or visual shortcut rule.\n"
            "   ⏱️ STANDARD VS SHORTCUT: Clear comparison showing why standard method takes 3 mins and shortcut takes 15 seconds.\n"
            "   🔍 OPTION ELIMINATION: How to eliminate 2 wrong options instantly using dimensions, symmetry, or boundary limits.\n"
            "   🎯 ACTIVE CHALLENGE: 1 quick practice problem testing their speed with this trick.\n"
            "2. ALWAYS COMPLETE ALL 4 SECTIONS FULLY. Never terminate mid-sentence.\n"
            "3. NO ASTERISKS (*) and NO DOLLAR SIGNS ($). Keep all equations in clean plain text notation."
        )
        context_str = ""

    elif is_compare_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — Senior IIT-JEE & NEET Professor.\n"
            "The student wants to compare or distinguish two related concepts (e.g. SN1 vs SN2, Isothermal vs Adiabatic).\n\n"
            "STRICT RULES:\n"
            "1. Structure your response clearly:\n"
            "   📌 CORE DEFINITIONS: Side-by-side core distinction.\n"
            "   ⚖️ KEY DIFFERENCES: Distinct comparison points (Mechanism, Equations, Conditions, Rate laws).\n"
            "   ⚠️ #1 EXAM CONFUSION / TRAP: The exact trap examiners use to test students on this pair.\n"
            "   💡 SOCRATIC CHECK: 1 practice question to test if they can identify which concept applies.\n"
            "2. Always write complete thoughts. Never terminate mid-sentence.\n"
            "3. NO ASTERISKS (*) or DOLLAR SIGNS ($)."
        )
        context_str = ""

    elif is_mistakes_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — Negative Marking Specialist for IIT-JEE & NEET.\n"
            "The student wants to know COMMON EXAM MISTAKES, TRAPS, and BLUNDERS.\n\n"
            "STRICT RULES:\n"
            "1. Present 3 to 4 dangerous negative-marking traps for this topic:\n"
            "   • Trap #1: The Common Misconception ➔ The True Scientific Reality ➔ Examiner Trap Phrase.\n"
            "   • Trap #2: Calculation / Sign Convention Blunder.\n"
            "   • Trap #3: Edge Case / Boundary Condition Trap.\n"
            "2. End with 1 Socratic check testing if they can spot the trap in a sample question.\n"
            "3. NO ASTERISKS (*) or DOLLAR SIGNS ($). Never terminate mid-sentence."
        )
        context_str = ""

    elif is_mindmap_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — an elite IIT-JEE, TG EAPCET, and NEET mentor.\n"
            "The student wants a visual concept MIND MAP / FLOWCHART for their topic.\n\n"
            "STRICT RULES:\n"
            "1. Output a compact, mobile-friendly ASCII box diagram (max 40 characters wide) or structured tree:\n"
            "   • [ Core Principle ] ➔ Fundamental definition\n"
            "   • [ Governing Formula ] ➔ Key equation in plain text\n"
            "   • [ High-Yield Branches ] ➔ Main sub-applications\n"
            "   • [ 10-Year Exam Trap ] ➔ Common misconception to avoid\n"
            "2. End with 1 sharp Socratic follow-up question.\n"
            "3. NO ASTERISKS (*) or DOLLAR SIGNS ($). Never cut off mid-sentence."
        )
        context_str = ""

    elif is_list_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering and medical entrance tutor.\n"
            "The student is asking for a list of high-yield practice questions / PYQ topics.\n\n"
            "STRICT RULES:\n"
            "1. Extract the number requested (e.g. top 5, top 10). If unspecified, provide 5 distinct questions.\n"
            "2. Cover DIFFERENT high-yield 10-year PYQ chapters across the syllabus.\n"
            "3. Format as a clean numbered list (1., 2., 3., ...). For each item include:\n"
            "   • Chapter & Exam Context (e.g. NEET 2024 / JEE Main 2023)\n"
            "   • Core Concept & Governing Formula\n"
            "   • Tactical Socratic hint for solving it\n"
            "4. Always output all requested questions completely. Never terminate mid-sentence.\n"
            "5. NO ASTERISKS (*) or DOLLAR SIGNS ($)."
        )
        context_str = ""

    elif is_biology_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — senior NEET Biology mentor specialized in Botany and Zoology.\n\n"
            "STRICT RULES:\n"
            "1. Deliver a rich, NCERT-grounded explanation tailored specifically for Biology aspirants in 4-5 clear points:\n"
            "   1. Core Biological Concept & Distinction\n"
            "   2. NCERT Classification, Cellular/Organ Structure, or Pathway\n"
            "   3. 10-Year NEET PYQ Weightage & Key Chapters\n"
            "   4. Common Assertion-Reason Trap / Exception\n"
            "   5. Active-recall Socratic Question\n"
            "2. Never write 'Mathematical formulation: N/A' or force physics headings on biology!\n"
            "3. Complete all points fully. NO ASTERISKS (*) or DOLLAR SIGNS ($)."
        )

    elif is_proof_request:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — a senior engineering entrance tutor for IIT-JEE (Main/Adv) and TG EAPCET.\n\n"
            "STRICT RULES:\n"
            "1. Provide a complete, mathematically rigorous 5-step derivation:\n"
            "   1. Physical Law / Fundamental Axiom\n"
            "   2. Calculus Setup & Initial Conditions\n"
            "   3. Step-by-Step Algebraic / Calculus Manipulation\n"
            "   4. Final Result & 10-Year PYQ Application\n"
            "   5. Socratic Follow-up Challenge\n"
            "2. Write formulas in clean plain text (e.g. F12 = -F21, dp/dt = 0).\n"
            "3. Complete every step thoroughly without cutting off. NO ASTERISKS (*) or DOLLAR SIGNS ($)."
        )
        context_str = ""

    elif is_short_response:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — a senior engineering and medical entrance tutor.\n"
            "The student gave a short response (e.g. '1', '2', 'option A') following up on the previous discussion.\n\n"
            "STRICT RULES:\n"
            "1. Address the student's selected option or number directly in relation to the conversation context.\n"
            "2. Explain why it is correct or incorrect in 3 clear points with governing formulas.\n"
            "3. End with an encouraging follow-up calculation or next step.\n"
            "4. NO ASTERISKS (*) or DOLLAR SIGNS ($). Never output irrelevant generic proofs."
        )

    else:
        max_tokens = 2048
        system_prompt = (
            "You are ErManower — a legendary senior engineering & medical entrance tutor for IIT-JEE (Main/Adv), TG EAPCET, and NEET.\n\n"
            "STRICT RULES:\n"
            "1. Deliver a natural, pedagogically brilliant Socratic explanation in 4-5 numbered points:\n"
            "   1. Core Principle / Definition: Clear, concise explanation with real intuition.\n"
            "   2. Mathematical Formulation / Key Equations: Plain text formulas (e.g. sin(theta) = opp/hyp).\n"
            "   3. 10-Year PYQ Context / Exam Weightage: How this is tested in JEE/EAPCET/NEET.\n"
            "   4. Common Trap / Edge Case: What students often get wrong.\n"
            "   5. Socratic Next Step: An encouraging question prompting the student to solve the next step.\n"
            "2. Always complete every point and thought fully. Never terminate mid-sentence.\n"
            "3. NO ASTERISKS (*) and NO DOLLAR SIGNS ($). Keep formatting clean and readable on mobile."
        )

    user_content = f"Student Query:\n{student_input}{context_str}"

    final_output = ""

    # 1. Primary Engine: High-speed Groq LPU models
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            models_to_try = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
                "groq/compound"
            ]
            for model_name in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=0.3,
                        max_tokens=max_tokens,
                    )
                    final_output = response.choices[0].message.content.strip()
                    logger.info("Successfully generated response using Groq model: %s", model_name)
                    break
                except Exception as err:
                    logger.warning("Groq model %s failed: %s. Trying next...", model_name, err)
        except Exception as groq_err:
            logger.warning("Groq engine failed: %s. Falling back to Gemini...", groq_err)

    # 2. Secondary Resilient Engine: Google Gemini 2.5 / 3.6 Flash Fallback
    if not final_output:
        google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_api_key:
            import httpx
            gemini_models = ["gemini-2.5-flash", "gemini-3.6-flash"]
            for g_model in gemini_models:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={google_api_key}"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": user_content}
                                ]
                            }
                        ],
                        "systemInstruction": {
                            "parts": [
                                {"text": system_prompt}
                            ]
                        },
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 2048
                        }
                    }
                    with httpx.Client(timeout=25.0) as http_client:
                        resp = http_client.post(url, json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            final_output = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            logger.info("Successfully generated response using Google Gemini fallback: %s", g_model)
                            break
                except Exception as g_err:
                    logger.warning("Gemini model %s failed: %s", g_model, g_err)

    if not final_output:
        raise RuntimeError("All AI inference engines (Groq and Google Gemini) failed.")

    final_output = final_output.replace("*", "").replace("$", "")
    logger.info("Fast Socratic Tutor completed in single pass. Output length: %d chars", len(final_output))
    return final_output
