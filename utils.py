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

    models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    response = None
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
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
                max_tokens=1024,
            )
            break
        except Exception:
            continue

    if response is None:
        return VisionExtractionResult(raw_text=text, question_summary=text[:500])

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
    Analyse an uploaded question image with Gemini 1.5 Flash Vision API to extract
    raw text, LaTeX equations, detected subject, detected exam, and question summary.
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not set. Falling back to guidance text.")
        return VisionExtractionResult(
            raw_text="",
            question_summary="Photo received! Please type your question as text to receive Socratic guidance.",
        )

    import base64
    import httpx

    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={google_api_key}"

    prompt_text = (
        "You are an academic OCR vision assistant for Indian engineering/medical entrance exams (IIT-JEE, TG EAPCET, NEET, IPE Board).\n"
        "Transcribe all question text, mathematical expressions in LaTeX ($$...$$), detected subject (Maths, Physics, or Chemistry), "
        "detected exam (JEE_MAIN, JEE_ADVANCED, TG_EAPCET, or IPE_BOARD), and a 1-paragraph summary.\n\n"
        "Return a single raw valid JSON object matching this schema:\n"
        "{\n"
        '  "raw_text": "<transcribed text>",\n'
        '  "equations_latex": ["$$...$$"],\n'
        '  "diagrams": [],\n'
        '  "detected_subject": "<Maths|Physics|Chemistry|null>",\n'
        '  "detected_exam": "<JEE_MAIN|JEE_ADVANCED|TG_EAPCET|IPE_BOARD|null>",\n'
        '  "question_summary": "<summary>"\n'
        "}\n"
        "Do NOT use markdown triple backticks. Return raw JSON only."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data,
                        }
                    },
                ]
            }
        ]
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            if raw_text_out.startswith("```"):
                raw_text_out = raw_text_out.split("\n", 1)[1] if "\n" in raw_text_out else raw_text_out[3:]
                if raw_text_out.endswith("```"):
                    raw_text_out = raw_text_out[:-3]
                raw_text_out = raw_text_out.strip()

            return VisionExtractionResult.model_validate_json(raw_text_out)
    except Exception as exc:
        logger.error("Gemini Vision API call failed: %s", exc, exc_info=True)
        return VisionExtractionResult(
            raw_text="",
            question_summary="Image received! Please type your question text directly so I can guide you step-by-step.",
        )


# ---------------------------------------------------------------------------
# Comprehensive Visual Diagram & Concept Map Engine
# ---------------------------------------------------------------------------

import urllib.parse

def _make_quickchart_url(chart_dict: dict) -> str:
    """Serialize and URL-encode chart configuration for QuickChart.io."""
    json_str = json.dumps(chart_dict, separators=(",", ":"))
    return f"https://quickchart.io/chart?c={urllib.parse.quote(json_str)}&w=500&h=300&bkg=white"


def get_topic_diagram_info(query: str) -> Optional[tuple[str, str]]:
    """
    Return (diagram_url, caption) for concepts across Physics, Chemistry, Maths, and Biology.
    """
    if not query:
        return None

    q = query.lower()

    # 1. MATHS: Trigonometry & Waves
    if any(w in q for w in ["trigonometry", "trigo", "sin", "cos", "tan", "unit circle"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["0°", "30°", "45°", "60°", "90°", "180°", "270°", "360°"],
                "datasets": [
                    {"label": "sin(θ)", "data": [0, 0.5, 0.707, 0.866, 1.0, 0, -1.0, 0], "borderColor": "#2563eb", "fill": False},
                    {"label": "cos(θ)", "data": [1.0, 0.866, 0.707, 0.5, 0, -1.0, 0, 1.0], "borderColor": "#dc2626", "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Trigonometric Unit Wave: sin(θ) vs cos(θ)"}}
        }
        return _make_quickchart_url(chart), "📐 *Visual Concept Map: Trigonometric Functions & Unit Circle*"

    # 2. MATHS: Calculus & Parabolas
    elif any(w in q for w in ["calculus", "parabola", "derivative", "integral", "maxima", "minima", "quadratic"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["-3", "-2", "-1", "0", "1", "2", "3"],
                "datasets": [
                    {"label": "y = x² (Parabola)", "data": [9, 4, 1, 0, 1, 4, 9], "borderColor": "#9333ea", "fill": True, "backgroundColor": "rgba(147,51,234,0.1)"},
                    {"label": "dy/dx = 2x (Tangent Slope)", "data": [-6, -4, -2, 0, 2, 4, 6], "borderColor": "#16a34a", "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Parabolic Curve & Tangent Derivative Profile"}}
        }
        return _make_quickchart_url(chart), "📈 *Visual Calculus Map: Curve Trajectory & Derivative Tangent*"

    # 3. PHYSICS: Projectile Motion
    elif any(w in q for w in ["projectile", "trajectory", "range", "flight time", "projectile motion"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["0m", "10m", "20m", "30m", "40m (R/2)", "50m", "60m", "70m", "80m (Range)"],
                "datasets": [
                    {"label": "Parabolic Path (y vs x)", "data": [0, 7.5, 12, 14.5, 15, 14.5, 12, 7.5, 0], "borderColor": "#2563eb", "fill": True, "backgroundColor": "rgba(37,99,235,0.15)"}
                ]
            },
            "options": {"title": {"display": True, "text": "Projectile Flight Trajectory: Max Height H & Range R"}}
        }
        return _make_quickchart_url(chart), "🚀 *Visual Physics Map: Projectile Parabolic Trajectory*"

    # 4. PHYSICS: Rotational Dynamics & Moment of Inertia
    elif any(w in q for w in ["rotation", "moment of inertia", "rolling", "angular momentum", "torque"]):
        chart = {
            "type": "bar",
            "data": {
                "labels": ["Ring (1.0)", "Hollow Cyl (1.0)", "Solid Disc (0.5)", "Hollow Sphere (0.67)", "Solid Sphere (0.4)"],
                "datasets": [
                    {"label": "Moment of Inertia Ratio (I / mR²)", "data": [1.0, 1.0, 0.5, 0.67, 0.4], "backgroundColor": ["#ef4444", "#f97316", "#3b82f6", "#eab308", "#10b981"]}
                ]
            },
            "options": {"title": {"display": True, "text": "Moment of Inertia Ratios for Rolling Acceleration"}}
        }
        return _make_quickchart_url(chart), "🔄 *Visual Rotational Dynamics Map: Moment of Inertia Comparison*"

    # 5. PHYSICS: Simple Harmonic Motion (SHM) Energy
    elif any(w in q for w in ["shm", "harmonic", "pendulum", "oscillation", "spring"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["-A", "-A/2", "0 (Mean)", "+A/2", "+A"],
                "datasets": [
                    {"label": "Potential Energy U = ½kx²", "data": [10, 2.5, 0, 2.5, 10], "borderColor": "#ef4444", "fill": False},
                    {"label": "Kinetic Energy K = ½m(v²)", "data": [0, 7.5, 10, 7.5, 0], "borderColor": "#3b82f6", "fill": False},
                    {"label": "Total Mechanical Energy E", "data": [10, 10, 10, 10, 10], "borderColor": "#10b981", "borderDash": [5, 5], "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "SHM Energy Conservation: Kinetic vs Potential vs Total"}}
        }
        return _make_quickchart_url(chart), "⚡ *Visual Physics Map: SHM Energy Conservation Curve*"

    # 6. PHYSICS: Modern Physics - Photoelectric Effect
    elif any(w in q for w in ["photoelectric", "work function", "stopping potential", "photon", "de broglie"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["0", "ν0 (Threshold)", "1.5 ν0", "2.0 ν0", "2.5 ν0", "3.0 ν0"],
                "datasets": [
                    {"label": "Stopping Potential V0 (Volts)", "data": [0, 0, 1.0, 2.0, 3.0, 4.0], "borderColor": "#f59e0b", "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Einstein Photoelectric Curve: Stopping Potential vs Frequency (Slope = h/e)"}}
        }
        return _make_quickchart_url(chart), "🔬 *Visual Modern Physics Map: Photoelectric Effect & Work Function*"

    # 7. PHYSICS: Thermodynamics - P-V Carnot Cycle
    elif any(w in q for w in ["thermo", "pv", "carnot", "entropy", "adiabatic", "isothermal"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["V1 (P1,T_H)", "V2 (P2,T_H)", "V3 (P3,T_C)", "V4 (P4,T_C)", "V1"],
                "datasets": [
                    {"label": "Carnot Cycle (P vs V)", "data": [100, 50, 20, 40, 100], "borderColor": "#ef4444", "fill": True, "backgroundColor": "rgba(239,68,68,0.15)"}
                ]
            },
            "options": {"title": {"display": True, "text": "P-V Thermodynamic Indicator Diagram: 4-Stage Carnot Cycle"}}
        }
        return _make_quickchart_url(chart), "🔥 *Visual Thermodynamics Map: P-V Indicator & Carnot Engine*"

    # 8. PHYSICS: Optics & Lens Ray Tracing
    elif any(w in q for w in ["optics", "lens", "refraction", "mirror", "ray", "prism", "snell"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["-2F", "-F", "0 (Lens)", "+F", "+2F"],
                "datasets": [
                    {"label": "Incident / Refracted Ray Path", "data": [10, 5, 0, -5, -10], "borderColor": "#2563eb", "fill": False},
                    {"label": "Principal Axis", "data": [0, 0, 0, 0, 0], "borderColor": "#9ca3af", "borderDash": [4, 4], "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Convex Lens Ray Tracing & Focal Plane Conformance"}}
        }
        return _make_quickchart_url(chart), "🔭 *Visual Optics Map: Ray Tracing & Lens Formula Geometry*"

    # 9. PHYSICS: Circuits & Kirchhoff Laws
    elif any(w in q for w in ["circuit", "kirchhoff", "ohm", "resistor", "wheatstone", "current", "potentiometer"]):
        chart = {
            "type": "bar",
            "data": {
                "labels": ["Branch I1 (Loop 1)", "Branch I2 (Loop 2)", "Combined I3 = I1 + I2"],
                "datasets": [
                    {"label": "Current Distribution (Amperes)", "data": [2.0, 1.5, 3.5], "backgroundColor": ["#3b82f6", "#10b981", "#f59e0b"]}
                ]
            },
            "options": {"title": {"display": True, "text": "Kirchhoff Current Law (KCL) Junction Distribution"}}
        }
        return _make_quickchart_url(chart), "⚡ *Visual Circuit Map: Kirchhoff Loop & Branch Currents*"

    # 10. PHYSICS: Free Body Diagram Forces
    elif any(w in q for w in ["force", "motion", "block", "free body", "fbd", "friction", "newton"]):
        chart = {
            "type": "radar",
            "data": {
                "labels": ["F_applied (Horizontal)", "Normal Force (Up)", "Gravity m·g (Down)", "Friction f_k (Opposite)"],
                "datasets": [
                    {"label": "Force Vector Magnitude (N)", "data": [20, 49, 49, 10], "backgroundColor": "rgba(59,130,246,0.25)", "borderColor": "#2563eb"}
                ]
            },
            "options": {"title": {"display": True, "text": "Free Body Force Vector Resolution Diagram"}}
        }
        return _make_quickchart_url(chart), "⚖️ *Visual Physics Map: Free Body Force Vector Diagram*"

    # 11. CHEMISTRY: Coordination Chemistry & CFT Splitting
    elif any(w in q for w in ["coordination", "crystal field", "cft", "t2g", "eg", "ligand", "octahedral"]):
        chart = {
            "type": "bar",
            "data": {
                "labels": ["Degenerate 3d", "t2g (Lower Energy -0.4 Δo)", "eg (Higher Energy +0.6 Δo)"],
                "datasets": [
                    {"label": "Orbital Energy Level in Octahedral Field", "data": [0, -4, 6], "backgroundColor": ["#9ca3af", "#3b82f6", "#ef4444"]}
                ]
            },
            "options": {"title": {"display": True, "text": "Crystal Field Theory (CFT) Octahedral Splitting (Δo)"}}
        }
        return _make_quickchart_url(chart), "🧪 *Visual Chemistry Map: Crystal Field Orbital Splitting Diagram*"

    # 12. CHEMISTRY: Chemical Kinetics & Activation Energy
    elif any(w in q for w in ["kinetics", "activation energy", "arrhenius", "catalyst", "reaction rate"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["Reactants (A+B)", "Transition State (Uncatalyzed)", "Transition State (Catalyzed)", "Products (C+D)"],
                "datasets": [
                    {"label": "Potential Energy Profile (kJ/mol)", "data": [20, 85, 50, -10], "borderColor": "#ef4444", "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Arrhenius Reaction Coordinate: Activation Energy Ea & Catalyst Lowering"}}
        }
        return _make_quickchart_url(chart), "⚗️ *Visual Chemistry Map: Activation Energy & Reaction Pathway*"

    # 13. CHEMISTRY: Periodic Trends (Ionization Energy)
    elif any(w in q for w in ["periodic", "ionization energy", "electronegativity", "atomic radius"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["Li", "Be (2s²)", "B (2p¹)", "C", "N (2p³ half-filled)", "O", "F", "Ne"],
                "datasets": [
                    {"label": "1st Ionization Enthalpy (kJ/mol)", "data": [520, 899, 801, 1086, 1402, 1314, 1681, 2081], "borderColor": "#8b5cf6", "fill": False}
                ]
            },
            "options": {"title": {"display": True, "text": "Period 2 Ionization Enthalpy Anomaly (Be > B and N > O)"}}
        }
        return _make_quickchart_url(chart), "📊 *Visual Periodic Table Map: Ionization Energy Anomalies & Trends*"

    # 14. BIOLOGY: Cell Cycle & Mitosis
    elif any(w in q for w in ["cell cycle", "mitosis", "meiosis", "interphase"]):
        chart = {
            "type": "pie",
            "data": {
                "labels": ["G1 Phase (Growth ~40%)", "S Phase (DNA Synthesis ~30%)", "G2 Phase (Prep ~20%)", "M Phase (Mitosis ~10%)"],
                "datasets": [
                    {"data": [40, 30, 20, 10], "backgroundColor": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]}
                ]
            },
            "options": {"title": {"display": True, "text": "NCERT Cell Cycle Phase Duration Profile"}}
        }
        return _make_quickchart_url(chart), "🧬 *Visual Biology Map: Cell Cycle Phase Proportions*"

    # 15. BIOLOGY: Genetics & Mendelian Ratios
    elif any(w in q for w in ["genetics", "mendel", "dihybrid", "monohybrid", "punnett"]):
        chart = {
            "type": "bar",
            "data": {
                "labels": ["Round-Yellow", "Round-Green", "Wrinkled-Yellow", "Wrinkled-Green"],
                "datasets": [
                    {"label": "Dihybrid F2 Phenotypic Ratio (9:3:3:1)", "data": [9, 3, 3, 1], "backgroundColor": ["#eab308", "#10b981", "#f97316", "#ef4444"]}
                ]
            },
            "options": {"title": {"display": True, "text": "Mendelian Dihybrid F2 Cross Phenotypic Ratio"}}
        }
        return _make_quickchart_url(chart), "🌿 *Visual Genetics Map: Mendelian Dihybrid Ratio Chart*"

    # 16. BIOLOGY: Respiratory Volumes (NEET High-Yield)
    elif any(w in q for w in ["respiration", "lung", "vital capacity", "tidal volume", "nephron", "heart"]):
        chart = {
            "type": "bar",
            "data": {
                "labels": ["Tidal Vol (TV: 500mL)", "Insp Reserve (IRV: 2500mL)", "Exp Reserve (ERV: 1100mL)", "Residual Vol (RV: 1200mL)"],
                "datasets": [
                    {"label": "Lung Capacity Volumes (mL)", "data": [500, 2500, 1100, 1200], "backgroundColor": ["#06b6d4", "#3b82f6", "#8b5cf6", "#f43f5e"]}
                ]
            },
            "options": {"title": {"display": True, "text": "Human Pulmonary Volumes & Capacities (NCERT)"}}
        }
        return _make_quickchart_url(chart), "🫁 *Visual Biology Map: Human Lung Volumes & Capacities*"

    # 17. MATHS: 3D Vectors & Geometry
    elif any(w in q for w in ["vector", "dot product", "cross product", "3d", "coordinate"]):
        chart = {
            "type": "line",
            "data": {
                "labels": ["X-axis", "Y-axis", "Z-axis"],
                "datasets": [
                    {"label": "Vector A (2, 1, -1)", "data": [2, 1, -1], "borderColor": "#16a34a"},
                    {"label": "Vector B (1, -1, 1)", "data": [1, -1, 1], "borderColor": "#9333ea"}
                ]
            },
            "options": {"title": {"display": True, "text": "3D Vector Cartesian Component Projections"}}
        }
        return _make_quickchart_url(chart), "📐 *Visual Vector Map: 3D Coordinate Component Projections*"

    return None


def get_topic_diagram_url(query: str) -> Optional[str]:
    """Backward compatibility helper returning just the diagram URL."""
    info = get_topic_diagram_info(query)
    return info[0] if info else None

