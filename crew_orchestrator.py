"""
ErManower JEE Bot — Socratic Tutor Engine
==========================================
Ultra-low-latency Socratic Tutor using Groq's llama-3.3-70b-versatile model
with local NCERT/JEE RAG context retrieval.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from pydantic import BaseModel, Field

from database import search_knowledge_bank

logger = logging.getLogger(__name__)






# ═══════════════════════════════════════════════════════════════════════════
# Public Orchestration API
# ═══════════════════════════════════════════════════════════════════════════

def run_crew(student_input: str) -> str:
    """
    Execute the multi-agent crew pipeline or fall back to fast tutor for minimal latency.
    """
    return run_fast_tutor(student_input)


def run_fast_tutor(student_input: str) -> str:
    """
    Ultra-low-latency Socratic Tutor engine (~0.8s response time).
    Direct single-pass execution combining local NCERT/JEE RAG context retrieval
    with Groq's llama-3.3-70b-versatile model.
    """
    logger.info("Executing Fast Socratic Tutor for input: %s", student_input[:100])

    # 1. Instant local RAG search (< 5ms)
    rag_results = search_knowledge_bank(query=student_input, top_k=3)
    context_str = ""
    if rag_results:
        # Only include reference docs that actually matched the query keywords (score > 0.5)
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

    is_list_request = any(
        phrase in student_input.lower()
        for phrase in ["top 5", "5 questions", "5 phy", "5 chem", "5 math", "list questions", "important questions", "top questions", "practice questions", "top neet", "top jee"]
    )
    is_proof_request = any(
        w in student_input.lower()
        for w in ["proof", "prove", "derive", "derivation", "law of motion", "explain momentum"]
    )
    is_short_response = student_input.strip().lower() in ["1", "2", "3", "4", "5", "a", "b", "c", "d", "option 1", "option 2", "option 3", "option 4", "option 5"]

    if is_proof_request:
        system_prompt = (
            "You are ErManower — a senior engineering tutor for IIT-JEE (Main/Adv), TG EAPCET, and Telangana IPE Board.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. RIGOROUS PROOF MODE: Provide a complete, mathematically sound 5-step derivation for the student's request.\n"
            "2. NO ASTERISKS: Do NOT use asterisks (*) or stars anywhere.\n"
            "3. NO DOLLAR SIGNS: Write all math formulas in clean plain text notation (e.g., F12 = -F21, dp/dt = 0, F = m · a).\n"
            "4. STEP-BY-STEP PROOF:\n"
            "   Point 1: Fundamental Axiom / Physical Principle (e.g., Conservation of Momentum or Newton's Law).\n"
            "   Point 2: Mathematical formulation & calculus setup.\n"
            "   Point 3: Step-by-step algebraic or calculus derivation.\n"
            "   Point 4: 10-Year PYQ application (e.g. recoil velocity, rocket motion, elastic collisions in TG EAPCET/JEE).\n"
            "   Point 5: Socratic follow-up question for the student's next step.\n"
            "5. CONCISENESS: Keep the output under 140 words in 5 clear numbered points (1., 2., 3., 4., 5.)."
        )
        context_str = ""
    elif is_short_response:
        system_prompt = (
            "You are ErManower — a senior engineering tutor for IIT-JEE and TG EAPCET.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. EVALUATE CHOICE: Acknowledge the student's selected option/number directly in point 1.\n"
            "2. FORMAT: Write EXACTLY 5 numbered points (1., 2., 3., 4., 5.). No asterisks (*) or stars.\n"
            "3. EXPLAIN: Explain the concept associated with their choice using 10-year PYQ insights in points 2-4.\n"
            "4. NO DOLLAR SIGNS: Write formulas in clean plain text notation.\n"
            "5. SOCRATIC QUESTION: End point 5 with a clear follow-up calculation or concept question."
        )
        context_str = ""
    elif is_list_request:
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering entrance tutor for IIT-JEE (Main/Adv), TG EAPCET, and NEET.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. 10-YEAR PYQ FOCUS: Select questions ONLY from top 10-year high-yield chapters (Physics: Modern Physics, Electrostatics, Optics, Current Electricity; Chemistry: GOC, Equilibrium, Coordination Compounds, Kinetics; Maths: Calculus, Vectors/3D, Matrices, Quadratic Equations).\n"
            "2. FORMAT: Write EXACTLY 5 numbered points (1., 2., 3., 4., 5.) covering 5 DIFFERENT high-yield 10-year PYQ topics/questions.\n"
            "3. NO ASTERISKS: Do NOT use asterisks (*) or stars anywhere.\n"
            "4. NO DOLLAR SIGNS: Write formulas in clean plain text notation.\n"
            "5. SOCRATIC HINT: For each question, state the key 10-year PYQ concept/formula and give a short guided hint.\n"
            "6. CONCISENESS: Keep the entire output crisp and under 120 words total."
        )
        context_str = ""  # Let LLM span multiple chapters freely for broad lists
    else:
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering entrance tutor for IIT-JEE (Main/Adv), TG EAPCET, and Telangana IPE Board.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. FORMAT: Write ONLY in numbered points (1., 2., 3., 4., 5.). Do NOT use asterisks (*) or stars anywhere.\n"
            "2. DYNAMIC CONTENT: Tailor the 5 points directly to the student's specific question. If they ask for a proof, explanation, or derivation, provide the step-by-step proof/derivation logic in points 1 to 4 and a Socratic check in point 5. Do NOT use fixed static headers.\n"
            "3. NO DOLLAR SIGNS: Never use dollar signs ($ or $$). Write all math formulas in clean plain text notation (e.g., F = dp/dt = m · a).\n"
            "4. 10-YEAR PYQ CONTEXT: Integrate 10-year PYQ weightage or exam tips for Hyderabad/Telangana engineering aspirants naturally into the explanation.\n"
            "5. SOCRATIC ENDING: Point 5 must always be a short, encouraging follow-up question testing their understanding of the next step.\n"
            "6. CONCISENESS: Keep the entire output crisp and under 120 words total."
        )

    user_content = f"Student Query:\n{student_input}{context_str}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=350,
    )

    final_output = response.choices[0].message.content.strip()
    final_output = final_output.replace("*", "").replace("$", "")
    logger.info("Fast Socratic Tutor completed in single pass. Output length: %d chars", len(final_output))
    return final_output
