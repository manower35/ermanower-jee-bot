"""
ErManower JEE Bot — Multimodal Text Query Analyser (Groq)
======================================================
Connects to Groq's Llama model for fast text analysis.
Detects subject, exam, equations, and question context from
student text messages. Returns a structured Pydantic payload.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from groq import Groq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------

class DiagramContext(BaseModel):
    """Structured representation of a diagram described in text."""
    diagram_type: str = Field(
        ...,
        description="Category: circuit, graph, geometric_figure, free_body_diagram, organic_structure, other",
    )
    description: str = Field(
        ...,
        description="Concise natural-language description of the diagram.",
    )


class VisionExtractionResult(BaseModel):
    """Structured extraction from a student query."""
    raw_text: str = Field(default="", description="Original student text.")
    equations_latex: list[str] = Field(
        default_factory=list,
        description="Mathematical equations transcribed into block LaTeX ($$...$$).",
    )
    diagrams: list[DiagramContext] = Field(
        default_factory=list,
        description="List of diagrams described.",
    )
    detected_subject: Optional[str] = Field(
        default=None,
        description="Best-guess academic subject: Maths, Physics, or Chemistry.",
    )
    detected_exam: Optional[str] = Field(
        default=None,
        description="Best-guess target exam: JEE_MAIN, JEE_ADVANCED, TG_EAPCET, or IPE_BOARD.",
    )
    question_summary: str = Field(
        default="",
        description="One-paragraph summary of the question.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = """\
You are a precise academic text analysis engine specialized in Indian engineering \
entrance exam content (IIT-JEE, TG EAPCET, IPE Board).

### Instructions
1. **Text Extraction**: Reproduce the student's query verbatim in `raw_text`.
2. **Equation Transcription**: Identify every mathematical expression or formula. \
   Transcribe each into flawless block LaTeX wrapped in $$ delimiters.
3. **Subject Detection**: Infer the academic subject (Maths, Physics, or Chemistry).
4. **Exam Detection**: Infer the likely target exam (JEE_MAIN, JEE_ADVANCED, \
   TG_EAPCET, or IPE_BOARD) from question style and difficulty.
5. **Question Summary**: Write a one-paragraph summary of the problem.

### Output Format
Return a single valid JSON object matching this schema exactly:
{
  "raw_text": "<string>",
  "equations_latex": ["$$...$$", ...],
  "diagrams": [],
  "detected_subject": "<Maths|Physics|Chemistry|null>",
  "detected_exam": "<JEE_MAIN|JEE_ADVANCED|TG_EAPCET|IPE_BOARD|null>",
  "question_summary": "<string>"
}
Do NOT wrap the JSON in markdown code fences. Return raw JSON only.
"""


# ---------------------------------------------------------------------------
# Groq client singleton
# ---------------------------------------------------------------------------

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy-initialise and return a Groq client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_text_query(text: str) -> VisionExtractionResult:
    """
    Analyse a plain-text student query with Groq to detect subject,
    exam context, and any inline equations.

    Parameters
    ----------
    text : str
        The raw text message from the student.

    Returns
    -------
    VisionExtractionResult
        Structured extraction result.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "A student sent the following text query about an engineering "
                    "entrance exam problem. Analyse it and return structured JSON.\n\n"
                    f"Student query:\n{text}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw_json = response.choices[0].message.content.strip()
    logger.debug("Groq analysis raw response: %s", raw_json)

    # Clean markdown fences if model wraps them anyway
    if raw_json.startswith("```"):
        raw_json = raw_json.split("\n", 1)[1] if "\n" in raw_json else raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
        raw_json = raw_json.strip()

    try:
        result = VisionExtractionResult.model_validate_json(raw_json)
    except Exception:
        logger.warning("Failed to parse structured JSON from Groq; wrapping raw text.")
        result = VisionExtractionResult(
            raw_text=text,
            question_summary=text[:500],
        )

    return result


def parse_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> VisionExtractionResult:
    """
    Placeholder for image parsing — Groq does not support vision.
    Returns a result asking the student to type their question instead.
    """
    logger.warning("Image parsing called but Groq does not support vision. Returning guidance.")
    return VisionExtractionResult(
        raw_text="",
        question_summary="Image received but photo analysis is not available. Please type your question as text.",
        detected_subject=None,
        detected_exam=None,
    )
