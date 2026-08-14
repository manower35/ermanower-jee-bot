"""
ErManower JEE Bot — Socratic Tutor Engine
==========================================
Ultra-low-latency Socratic Tutor using Groq's llama-3.3-70b-versatile model
with local NCERT/JEE RAG context retrieval and intelligent multi-mode routing.
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
    Ultra-low-latency Socratic Tutor engine (~0.8s response time).
    Direct single-pass execution combining local NCERT/JEE RAG context retrieval
    with Groq's llama-3.3-70b-versatile model and dynamic intent handling.
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
    is_mindmap_request = bool(re.search(r'\b(mind\s*map|mindmap|diagram|flowchart|chart|graph|draw|sketch|map)\b', text_lower))
    is_list_request = bool(re.search(r'\b(top\s*\d+|list|\d+\s*questions|\d+\s*pyqs|important\s*questions|practice\s*questions|question\s*bank|top\s*neet|top\s*jee)\b', text_lower))
    is_proof_request = bool(re.search(r'\b(proof|prove|derive|derivation|step\s*by\s*step\s*derivation|proof\s*of)\b', text_lower))
    is_biology_request = bool(re.search(r'\b(botany|zoology|biology|neet\s*bio|cell|genetics|photosynthesis|plant\s*kingdom|animal\s*kingdom|human\s*physiology|biomolecules|ecology|morphology)\b', text_lower))
    is_short_response = text_clean in [
        "1", "2", "3", "4", "5", "a", "b", "c", "d",
        "option 1", "option 2", "option 3", "option 4",
        "option a", "option b", "option c", "option d"
    ]

    max_tokens = 500

    if is_mindmap_request:
        max_tokens = 600
        system_prompt = (
            "You are ErManower — an elite IIT-JEE, TG EAPCET, and NEET mentor.\n"
            "The student wants a visual concept MIND MAP / FLOWCHART for their topic.\n\n"
            "STRICT RULES:\n"
            "1. Output a structured, beautiful ASCII text mind map box using box-drawing characters (┌, ─, ┐, │, └, ┘, ──▶).\n"
            "2. The mind map MUST break down the topic into:\n"
            "   • [ Core Principle ] ──▶ Fundamental definition / law\n"
            "   • [ Governing Formula ] ──▶ Key equation in plain text\n"
            "   • [ High-Yield Branches ] ──▶ Main applications (e.g. collisions, rocket motion, circular motion)\n"
            "   • [ 10-Year PYQ Trap ] ──▶ Common exam pitfall / misconception\n"
            "3. End with 1 sharp Socratic follow-up question testing the student on their next calculation step.\n"
            "4. NO ASTERISKS (*) or DOLLAR SIGNS ($). Keep all text clean and readable on mobile Telegram."
        )
        context_str = ""

    elif is_list_request:
        max_tokens = 900
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering and medical entrance tutor (IIT-JEE / NEET / TG EAPCET).\n"
            "The student is asking for a list of high-yield practice questions / PYQ topics.\n\n"
            "STRICT RULES:\n"
            "1. Extract the number requested (e.g., top 5, top 10). If unspecified, provide 5 distinct questions.\n"
            "2. Cover DIFFERENT high-yield 10-year PYQ chapters across the syllabus (e.g., Modern Physics, Electrostatics, Ray Optics, Thermodynamics, GOC, Coordination Compounds, Calculus, Vectors, Genetics, Physiology).\n"
            "3. Format as a clean numbered list (1., 2., 3., ...). For each item include:\n"
            "   • Chapter & Exam Context (e.g. NEET 2024 / JEE Main)\n"
            "   • Core Question Concept & Governing Formula\n"
            "   • A tactical Socratic hint for solving it\n"
            "4. NO ASTERISKS (*) or DOLLAR SIGNS ($). Write formulas in clean plain text (e.g. F = m · a, lambda = h/p)."
        )
        context_str = ""

    elif is_biology_request:
        system_prompt = (
            "You are ErManower — senior NEET Biology mentor specialized in Botany and Zoology.\n\n"
            "STRICT RULES:\n"
            "1. Deliver a rich, NCERT-grounded explanation tailored specifically for Biology aspirants.\n"
            "2. Structure your response in 4 to 5 clear numbered points:\n"
            "   • Point 1: Fundamental biological concept & core distinction\n"
            "   • Point 2: Key NCERT classification, organ/cell structure, or physiological pathway\n"
            "   • Point 3: 10-Year NEET PYQ weightage & high-yield chapter context\n"
            "   • Point 4: Common NEET Assertion-Reason trap or exception to remember\n"
            "   • Point 5: Socratic active-recall question testing their conceptual mastery\n"
            "3. Do NOT write 'Mathematical formulation: N/A' or force physics headings on biology!\n"
            "4. NO ASTERISKS (*) or DOLLAR SIGNS ($)."
        )

    elif is_proof_request:
        system_prompt = (
            "You are ErManower — a senior engineering entrance tutor for IIT-JEE (Main/Adv) and TG EAPCET.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. RIGOROUS PROOF MODE: Provide a complete, mathematically sound 5-step derivation.\n"
            "2. Structure in 5 clear numbered points:\n"
            "   Point 1: Physical Law / Fundamental Axiom (e.g., Conservation of Momentum, Gauss Law).\n"
            "   Point 2: Mathematical formulation & setup (e.g., dp/dt = 0, F_net = 0).\n"
            "   Point 3: Step-by-step algebraic or calculus derivation.\n"
            "   Point 4: 10-Year PYQ application in JEE / EAPCET.\n"
            "   Point 5: Socratic follow-up question for the student's next step.\n"
            "3. NO ASTERISKS (*) and NO DOLLAR SIGNS ($). Write formulas in clean plain text (e.g. F12 = -F21).\n"
            "4. Keep math rigorous, accurate, and under 140 words."
        )
        context_str = ""

    elif is_short_response:
        system_prompt = (
            "You are ErManower — a senior engineering and medical entrance tutor.\n"
            "The student gave a short response (e.g. '1', '2', 'option A') following up on the previous discussion.\n\n"
            "STRICT RULES:\n"
            "1. Address the student's selected option or number directly in the context of the previous conversation.\n"
            "2. Explain why it is correct or incorrect using 10-year PYQ insights in 3-4 crisp points.\n"
            "3. State any relevant governing formula in clean plain text notation.\n"
            "4. End with an encouraging follow-up calculation or concept check.\n"
            "5. NO ASTERISKS (*) or DOLLAR SIGNS ($). Do NOT output irrelevant generic proofs."
        )

    else:
        system_prompt = (
            "You are ErManower — a legendary senior engineering & medical entrance tutor for IIT-JEE (Main/Adv), TG EAPCET, and NEET.\n\n"
            "STRICT RULES:\n"
            "1. Deliver a natural, pedagogically brilliant Socratic explanation in 4-5 numbered points tailored directly to the student's query.\n"
            "2. State the governing principle, definition, or theorem clearly in point 1.\n"
            "3. Write all formulas in clean plain text notation (e.g., F = G · m1 · m2 / r^2, E = h · nu - phi).\n"
            "4. Naturally weave in 10-year PYQ weightage, exam trends, or common traps for Indian engineering/medical aspirants.\n"
            "5. Point 5 must ALWAYS be an encouraging Socratic question prompting the student to execute the next calculation or reasoning step.\n"
            "6. NO ASTERISKS (*) and NO DOLLAR SIGNS ($). Avoid robotic repetitive headers."
        )

    user_content = f"Student Query:\n{student_input}{context_str}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    final_output = response.choices[0].message.content.strip()
    final_output = final_output.replace("*", "").replace("$", "")
    logger.info("Fast Socratic Tutor completed in single pass. Output length: %d chars", len(final_output))
    return final_output
