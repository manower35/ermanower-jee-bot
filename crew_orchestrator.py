"""
ErManower JEE Bot — CrewAI Multi-Agent Orchestrator
===================================================
Assembles a sequential three-agent crew:
  1. Exam Context Analyst — tags subject and exam from student input.
  2. Knowledge Bank Retriever — searches the Chroma vector store.
  3. Socratic Senior Tutor — synthesizes a guided, hint-based response.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from database import search_knowledge_bank

import litellm
litellm.drop_params = True
_orig_completion = litellm.completion

def _clean_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                msg.pop("cache_control", None)
    return _orig_completion(*args, **kwargs)

litellm.completion = _clean_completion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configure CrewAI to use Groq Llama 3.3 70B (free, ultra-fast)
# ---------------------------------------------------------------------------
_groq_llm = "groq/llama-3.3-70b-versatile"



# ═══════════════════════════════════════════════════════════════════════════
# CrewAI Tool — Knowledge Bank Search
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeBankSearchInput(BaseModel):
    """Input schema for the knowledge bank search tool."""
    query: str = Field(..., description="Natural language search query about the student's problem.")
    exam: Optional[str] = Field(
        default=None,
        description="Exam filter: JEE_MAIN, JEE_ADVANCED, TG_EAPCET, or IPE_BOARD.",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject filter: Maths, Physics, or Chemistry.",
    )


@tool("knowledge_bank_search")
def knowledge_bank_search_tool(
    query: str,
    exam: str = "",
    subject: str = "",
) -> str:
    """
    Search the ErManower knowledge bank for relevant formulas, textbook notes,
    and previous year questions (PYQs). Apply exam and subject filters to
    narrow results before semantic matching.
    """
    results = search_knowledge_bank(
        query=query,
        exam=exam if exam else None,
        subject=subject if subject else None,
        top_k=5,
    )

    if not results:
        return json.dumps({"results": [], "message": "No matching documents found in the knowledge bank."})

    return json.dumps({"results": results}, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Agent Definitions
# ═══════════════════════════════════════════════════════════════════════════

_EXAM_CONTEXT_ANALYST = Agent(
    role="Exam Context Analyst",
    goal=(
        "Analyze the student's query (text and/or vision extraction output) to "
        "identify the exact academic subject (Maths, Physics, or Chemistry) and "
        "the target exam standard (JEE_MAIN, JEE_ADVANCED, TG_EAPCET, or IPE_BOARD). "
        "Produce a concise context tag that downstream agents can use for retrieval "
        "and response calibration."
    ),
    backstory=(
        "You are a seasoned Indian engineering entrance exam analyst with deep "
        "knowledge of IIT-JEE (Main & Advanced), Telangana EAPCET, and Intermediate "
        "Public Examinations (IPE) Board curricula. You can instantly recognize "
        "question patterns, syllabus boundaries, difficulty tiers, and exam-specific "
        "formatting conventions from minimal input cues."
    ),
    verbose=False,
    allow_delegation=False,
    llm=_groq_llm,
)

_KNOWLEDGE_BANK_RETRIEVER = Agent(
    role="Knowledge Bank Retriever",
    goal=(
        "Using the subject and exam tags provided by the Exam Context Analyst, "
        "search the vector knowledge bank to retrieve the most relevant formulas, "
        "theorems, textbook excerpts, and previous year questions (PYQs) that "
        "directly address the student's problem."
    ),
    backstory=(
        "You are a precision-oriented research assistant specialized in JEE and "
        "EAPCET academic databases. You use metadata-prefiltered semantic search "
        "to find the exact reference material a student needs. You always include "
        "source metadata (exam year, question number) when available."
    ),
    tools=[knowledge_bank_search_tool],
    verbose=False,
    allow_delegation=False,
    llm=_groq_llm,
)

_SOCRATIC_SENIOR_TUTOR = Agent(
    role="Socratic Senior Tutor",
    goal=(
        "Synthesize a pedagogically excellent student response that teaches through "
        "guided discovery. NEVER reveal the final answer or option directly. Instead: "
        "(1) State the governing physical/mathematical/chemical principle, "
        "(2) Display all relevant formulas using block LaTeX ($$...$$), "
        "(3) Present a tactical hint that prompts the student for their next step, "
        "(4) If PYQ data is available, mention the exam year and question context."
    ),
    backstory=(
        "You are ErManower — a legendary senior JEE tutor renowned across Telangana "
        "and Andhra Pradesh for your Socratic teaching style. You believe students "
        "learn best when they derive insights themselves. You never spoon-feed "
        "answers; instead, you illuminate the path with principles, formulas, and "
        "strategic hints. Your explanations are crisp, LaTeX-formatted, and deeply "
        "rooted in NCERT/JEE-level rigor. You address students warmly and "
        "encourage them to take the next reasoning step."
    ),
    verbose=False,
    allow_delegation=False,
    llm=_groq_llm,
)


# ═══════════════════════════════════════════════════════════════════════════
# Task Definitions Factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_tasks(student_input: str) -> list[Task]:
    """
    Construct the sequential task chain for the crew.

    Parameters
    ----------
    student_input : str
        Combined textual representation of the student query, including any
        vision extraction JSON.
    """
    task_analyse = Task(
        description=(
            f"Analyze the following student input and identify:\n"
            f"1. The academic subject (Maths / Physics / Chemistry)\n"
            f"2. The target exam (JEE_MAIN / JEE_ADVANCED / TG_EAPCET / IPE_BOARD)\n"
            f"3. A concise one-line summary of the core question\n\n"
            f"--- STUDENT INPUT ---\n{student_input}\n--- END ---\n\n"
            f"Return your analysis as a structured block with fields: "
            f"subject, exam, question_summary."
        ),
        expected_output=(
            "A structured analysis containing: subject (Maths/Physics/Chemistry), "
            "exam (JEE_MAIN/JEE_ADVANCED/TG_EAPCET/IPE_BOARD), and a concise "
            "question_summary string."
        ),
        agent=_EXAM_CONTEXT_ANALYST,
    )

    task_retrieve = Task(
        description=(
            "Using the subject and exam context from the previous analysis, "
            "call the knowledge_bank_search tool to retrieve up to 5 relevant "
            "documents including formulas, theorems, textbook notes, and PYQs. "
            "Pass the question_summary as the query string, and the detected "
            "subject and exam as filter parameters.\n\n"
            "Compile all retrieved content into a structured reference block "
            "that the tutor agent can use."
        ),
        expected_output=(
            "A compiled reference block containing all retrieved formulas, "
            "theorems, textbook notes, and PYQ references with their source "
            "metadata and relevance scores."
        ),
        agent=_KNOWLEDGE_BANK_RETRIEVER,
    )

    task_tutor = Task(
        description=(
            "You are responding to a student preparing for an Indian engineering "
            "entrance exam. Using the context analysis and retrieved reference "
            "material from previous agents, compose your Socratic teaching response.\n\n"
            "RULES:\n"
            "- NEVER give the final answer or option directly.\n"
            "- State the governing principle/theorem/law clearly.\n"
            "- Display ALL relevant formulas using block LaTeX: $$formula$$\n"
            "- Provide a tactical hint that guides the student to the next step.\n"
            "- If PYQ data is available, mention the exam year and question context.\n"
            "- End with an encouraging prompt asking the student what they think "
            "the next step should be.\n"
            "- Keep the response clear, structured, and under 600 words.\n"
            "- Use Telegram-compatible formatting (Markdown V2 is NOT used; "
            "plain text with LaTeX blocks is preferred).\n\n"
            f"--- ORIGINAL STUDENT INPUT ---\n{student_input}\n--- END ---"
        ),
        expected_output=(
            "A Socratic teaching response with: governing principle, LaTeX-formatted "
            "formulas ($$...$$), a tactical hint, optional PYQ reference, and an "
            "encouraging closing prompt for the student."
        ),
        agent=_SOCRATIC_SENIOR_TUTOR,
    )

    return [task_analyse, task_retrieve, task_tutor]


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

    if is_list_request:
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering entrance tutor for IIT-JEE, TG EAPCET, and NEET.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. FORMAT: Write EXACTLY 5 numbered points (1., 2., 3., 4., 5.) covering 5 DIFFERENT high-yield topics/questions across the subject (e.g., Mechanics, Electromagnetism, Optics, Modern Physics, Thermodynamics).\n"
            "2. NO ASTERISKS: Do NOT use asterisks (*) or stars anywhere.\n"
            "3. NO DOLLAR SIGNS: Write formulas in clean plain text notation.\n"
            "4. SOCRATIC HINT: For each question, state the key concept/formula and give a short guided hint.\n"
            "5. CONCISENESS: Keep the entire output crisp and under 120 words total."
        )
        context_str = ""  # Let LLM span multiple chapters freely for broad lists
    else:
        system_prompt = (
            "You are ErManower — a legendary Hyderabad senior engineering entrance tutor for IIT-JEE (Main/Adv), TG EAPCET, and Telangana IPE Board.\n\n"
            "STRICT FORMATTING RULES:\n"
            "1. FORMAT: Write ONLY in numbered points (1., 2., 3., 4., 5.). Do NOT use asterisks (*) or stars anywhere.\n"
            "2. NO DOLLAR SIGNS: Never use dollar signs ($ or $$). Write all math formulas in clean plain text notation (e.g., F = m · g, a = 9.8 m/s²).\n"
            "3. HYDERABAD CONTEXT: Tailor explanations for Hyderabad/Telangana engineering aspirants preparing for IIT-JEE, TG EAPCET, and IPE Board.\n"
            "4. SOCRATIC HINT: Never reveal the final answer directly. Provide a tactical hint and end with a question for their next step.\n"
            "5. CONCISENESS: Keep the entire output under 100 words in 5 clear points.\n\n"
            "EXAMPLE STRUCTURE:\n"
            "TOPIC: [Topic Name] (IIT-JEE / TG EAPCET)\n\n"
            "1. Key Concept: [1-line concept]\n"
            "2. Governing Formula: [Plain text formula without dollar signs]\n"
            "3. Exam Context: [Hyderabad/TG EAPCET tip]\n"
            "4. Tactical Hint: [Guided hint without final answer]\n"
            "5. Next Step: [Short question asking for next calculation step]"
        )

    user_content = f"Student Query:\n{student_input}{context_str}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=350,
    )

    final_output = response.choices[0].message.content.strip()
    final_output = final_output.replace("*", "").replace("$", "")
    logger.info("Fast Socratic Tutor completed in single pass. Output length: %d chars", len(final_output))
    return final_output
