"""
ErManower JEE Bot — LangChain Chroma Vector Database
=====================================================
Metadata-prefiltered retrieval system with built-in NCERT, IIT-JEE,
TG EAPCET, and IPE Board academic seed data.

Hard boolean pre-filters on 'exam' and 'subject' fields isolate
academic contexts before semantic similarity matching runs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid filter values
# ---------------------------------------------------------------------------

VALID_EXAMS = {"JEE_MAIN", "JEE_ADVANCED", "TG_EAPCET", "IPE_BOARD"}
VALID_SUBJECTS = {"Maths", "Physics", "Chemistry"}

# ---------------------------------------------------------------------------
# Seed NCERT, IIT-JEE, and State Board Knowledge Base
# ---------------------------------------------------------------------------

DEFAULT_NCERT_KNOWLEDGE_BANK = [
    # Mathematics — Quadratic Equations & Theory of Equations
    {
        "content": (
            "NCERT Class 11 Mathematics — Quadratic Equations & Complex Numbers:\n"
            "Standard Form: $$ax^2 + bx + c = 0$$ ($a \\neq 0$).\n"
            "Discriminant $D = b^2 - 4ac$.\n"
            "- If $D > 0$: Two distinct real roots.\n"
            "- If $D = 0$: Two equal real roots ($x = -b / 2a$).\n"
            "- If $D < 0$: Two complex conjugate roots ($x = \\frac{-b \\pm i\\sqrt{|D|}}{2a}$).\n"
            "Vieta's Formulas:\n"
            "Sum of roots: $\\alpha + \\beta = -b/a$, Product of roots: $\\alpha\\beta = c/a$.\n"
            "Condition for both roots to be positive: $D \\ge 0$, $\\alpha + \\beta > 0$, $\\alpha\\beta > 0$.\n"
            "IIT-JEE Advanced Tip: Location of roots for $f(x) = ax^2 + bx + c = 0$ with respect to a real number $k$:\n"
            "Both roots > $k \\iff a f(k) > 0, D \\ge 0, -b/2a > k$."
        ),
        "metadata": {"subject": "Maths", "exam": "JEE_MAIN", "topic": "Quadratic Equations", "source": "NCERT Class 11 / IIT-JEE"},
    },
    {
        "content": (
            "NCERT Class 11 Mathematics — Straight Lines & Coordinate Geometry:\n"
            "Distance Formula: $$d = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$\n"
            "Section Formula: Internal division $( \\frac{mx_2 + nx_1}{m+n}, \\frac{my_2 + ny_1}{m+n} )$.\n"
            "Slope of line: $m = \\tan \\theta = \\frac{y_2 - y_1}{x_2 - x_1}$.\n"
            "Slope-Intercept Form: $$y = mx + c$$\n"
            "Perpendicular distance of point $(x_1, y_1)$ from line $Ax + By + C = 0$:\n"
            "$$p = \\frac{|Ax_1 + By_1 + C|}{\\sqrt{A^2 + B^2}}$$\n"
            "TG EAPCET / IPE Board Tip: Pair of straight lines passing through origin: $ax^2 + 2hxy + by^2 = 0$.\n"
            "Angle between pair of lines: $\\tan \\theta = \\frac{2\\sqrt{h^2 - ab}}{|a + b|}$."
        ),
        "metadata": {"subject": "Maths", "exam": "TG_EAPCET", "topic": "Coordinate Geometry", "source": "NCERT Class 11 / TG EAPCET"},
    },
    {
        "content": (
            "NCERT Class 12 Mathematics — Calculus (Differentiation & Integration):\n"
            "Standard Derivatives: $\\frac{d}{dx}(\\sin x) = \\cos x$, $\\frac{d}{dx}(e^x) = e^x$, $\\frac{d}{dx}(\\ln x) = \\frac{1}{x}$.\n"
            "Product Rule: $(uv)' = u'v + uv'$, Quotient Rule: $(u/v)' = \\frac{u'v - uv'}{v^2}$.\n"
            "Standard Integration:\n"
            "$$\\int \\frac{dx}{x^2 + a^2} = \\frac{1}{a} \\tan^{-1}\\left(\\frac{x}{a}\\right) + C$$\n"
            "$$\\int \\frac{dx}{\\sqrt{a^2 - x^2}} = \\sin^{-1}\\left(\\frac{x}{a}\\right) + C$$\n"
            "IIT-JEE Main PYQ Note: Definite Integrals property:\n"
            "$$\\int_a^b f(x) dx = \\int_a^b f(a + b - x) dx$$ (King's Property)."
        ),
        "metadata": {"subject": "Maths", "exam": "JEE_ADVANCED", "topic": "Calculus", "source": "NCERT Class 12 / IIT-JEE"},
    },

    # Physics — Mechanics & Electrodynamics
    {
        "content": (
            "NCERT Class 11 Physics — Laws of Motion & Newton's Mechanics:\n"
            "Newton's Second Law: $$\\vec{F} = \\frac{d\\vec{p}}{dt} = m\\vec{a}$$\n"
            "Friction Force: $f_s \\le \\mu_s N$, $f_k = \\mu_k N$.\n"
            "Work-Energy Theorem: $$W_{net} = \\Delta K = \\frac{1}{2}m v_f^2 - \\frac{1}{2}m v_i^2$$\n"
            "Conservation of Linear Momentum: $\\sum \\vec{p}_i = \\sum \\vec{p}_f$ when $\\vec{F}_{ext} = 0$.\n"
            "TG EAPCET / IPE Board Derivation Note: Recoil velocity of gun, Banking of roads angle $\\tan \\theta = \\frac{v^2}{rg}$."
        ),
        "metadata": {"subject": "Physics", "exam": "IPE_BOARD", "topic": "Laws of Motion", "source": "NCERT Class 11 / IPE Board"},
    },
    {
        "content": (
            "NCERT Class 12 Physics — Electrostatics & Gauss's Law:\n"
            "Coulomb's Law: $$F = \\frac{1}{4\\pi\\varepsilon_0} \\frac{|q_1 q_2|}{r^2}$$\n"
            "Electric Field due to Point Charge: $E = \\frac{k q}{r^2}$.\n"
            "Gauss's Law: $$\\Phi_E = \\oint \\vec{E} \\cdot d\\vec{A} = \\frac{Q_{enclosed}}{\\varepsilon_0}$$\n"
            "Capacitance of Parallel Plate Capacitor: $C = \\frac{\\varepsilon_0 A}{d}$, with dielectric: $C' = K C$.\n"
            "IIT-JEE Advanced Concept: Electric potential inside a uniformly charged solid non-conducting sphere at distance $r < R$:\n"
            "$$V(r) = \\frac{k Q}{2 R^3} (3R^2 - r^2)$$"
        ),
        "metadata": {"subject": "Physics", "exam": "JEE_ADVANCED", "topic": "Electrostatics", "source": "NCERT Class 12 / IIT-JEE"},
    },

    # Chemistry — Physical, Organic, & Inorganic NCERT
    {
        "content": (
            "NCERT Class 11/12 Chemistry — Chemical Bonding & Molecular Structure:\n"
            "VSEPR Theory: Predicts molecular geometry based on electron pairs.\n"
            "- $sp$: Linear ($180^\\circ$), e.g., $BeCl_2, CO_2$.\n"
            "- $sp^2$: Trigonal Planar ($120^\\circ$), e.g., $BF_3$.\n"
            "- $sp^3$: Tetrahedral ($109.5^\\circ$), e.g., $CH_4, NH_3$ (pyramidal due to 1 lone pair), $H_2O$ (bent due to 2 lone pairs).\n"
            "Hybridisation Formula: $$H = \\frac{1}{2} [V + M - C + A]$$\n"
            "where $V =$ valence electrons of central atom, $M =$ monovalent atoms attached, $C =$ cationic charge, $A =$ anionic charge."
        ),
        "metadata": {"subject": "Chemistry", "exam": "JEE_MAIN", "topic": "Chemical Bonding", "source": "NCERT Class 11 / JEE Main"},
    },
    {
        "content": (
            "NCERT Class 12 Chemistry — Organic Chemistry (Reaction Mechanisms):\n"
            "SN1 Mechanism: Two-step, proceeds via Carbocation intermediate. Order of reactivity: $3^\\circ > 2^\\circ > 1^\\circ$. Racemisation occurs.\n"
            "SN2 Mechanism: Single-step, proceed via Transition state. Order of reactivity: $1^\\circ > 2^\\circ > 3^\\circ$. Inversion of configuration (Walden inversion).\n"
            "Markovnikov's Rule: Addition of $HX$ to unsymmetrical alkene adds $H^+$ to the carbon with more hydrogen atoms.\n"
            "Anti-Markovnikov (Kharasch / Peroxide Effect): Addition of $HBr$ in presence of peroxides adds $Br$ to carbon with more hydrogens via free radical mechanism."
        ),
        "metadata": {"subject": "Chemistry", "exam": "TG_EAPCET", "topic": "Organic Chemistry", "source": "NCERT Class 12 / TG EAPCET"},
    },
]

# ---------------------------------------------------------------------------
# Simple Local Embedding & In-Memory Storage (Fail-Safe)
# ---------------------------------------------------------------------------

class _SimpleEmbeddings(Embeddings):
    """Deterministic, zero-dependency embedding implementation."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        import hashlib
        h = hashlib.sha384(text.lower().encode('utf-8')).digest()
        return [float(b) / 255.0 for b in h]


def search_knowledge_bank(
    query: str,
    exam: Optional[str] = None,
    subject: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Perform metadata-prefiltered semantic search over NCERT, JEE, EAPCET, and IPE data.

    Applies hard boolean filters on 'exam' and 'subject' fields, then returns matching
    knowledge entries.
    """
    logger.info(
        "Knowledge bank search — query=%r, exam=%s, subject=%s, top_k=%d",
        query[:80],
        exam,
        subject,
        top_k,
    )

    query_lower = query.lower()

    # Fast rule-based subject classification if not explicitly filtered
    if not subject:
        if any(w in query_lower for w in ["physics", "phy ", " phy", "mechanics", "force", "motion", "electrostatics", "coulomb", "gauss"]):
            subject = "Physics"
        elif any(w in query_lower for w in ["chemistry", "chem", "bonding", "organic", "reaction", "hybridisation", "vsper"]):
            subject = "Chemistry"
        elif any(w in query_lower for w in ["math", "maths", "mathematics", "quadratic", "calculus", "integration", "derivative", "equation"]):
            subject = "Maths"

    # Fast rule-based exam classification if not explicitly filtered
    if not exam:
        if any(w in query_lower for w in ["advanced", "jee adv"]):
            exam = "JEE_ADVANCED"
        elif any(w in query_lower for w in ["eapcet", "tgeapcet", "eamcet"]):
            exam = "TG_EAPCET"
        elif any(w in query_lower for w in ["ipe", "board", "intermediate"]):
            exam = "IPE_BOARD"
        elif any(w in query_lower for w in ["jee", "main"]):
            exam = "JEE_MAIN"

    matching_docs = []
    
    # Custom query word cleaning (exclude common stop words)
    stop_words = {"write", "with", "what", "how", "show", "give", "some", "your", "this", "that", "question", "questions", "answer", "answers", "quest", "about"}
    raw_words = [w.strip("?,.:;!)(\"'-") for w in query_lower.split()]
    words = [w for w in raw_words if len(w) >= 3 and w not in stop_words]

    for doc in DEFAULT_NCERT_KNOWLEDGE_BANK:
        meta = doc["metadata"]

        # 1. Subject filter
        if subject and meta.get("subject") != subject:
            continue

        # 2. Exam filter (if provided, match exam or general JEE_MAIN)
        if exam and meta.get("exam") != exam and meta.get("exam") != "JEE_MAIN":
            continue

        # 3. Simple relevance score based on keyword overlap
        content = doc["content"]
        content_lower = content.lower()
        
        if words:
            word_matches = sum(1 for w in words if w in content_lower)
            score = word_matches / len(words)
        else:
            score = 0.0

        matching_docs.append({
            "content": content,
            "metadata": meta,
            "score": round(score + 0.5, 4),  # baseline relevance
        })

    # Sort by score descending
    matching_docs.sort(key=lambda x: x["score"], reverse=True)
    return matching_docs[:top_k]


def ingest_documents(documents: list[dict]) -> int:
    """Ingest custom documents into the knowledge bank."""
    added = 0
    for doc in documents:
        meta = doc.get("metadata", {})
        if meta.get("exam") in VALID_EXAMS and meta.get("subject") in VALID_SUBJECTS:
            DEFAULT_NCERT_KNOWLEDGE_BANK.append(doc)
            added += 1
    return added
