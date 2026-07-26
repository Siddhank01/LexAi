"""
Input validation guard — runs before any ML code.

Returns (True, "")            if valid.
Returns (False, error_msg)    if invalid.

Catches:
  1. Empty / whitespace only
  2. Too long  (> QUERY_MAX_CHARS)     — checked BEFORE word count
  3. Too short (< QUERY_MIN_CHARS)
  4. Too few words (< QUERY_MIN_WORDS)
  5. Non-Latin / Indic script
  6. No legal signal words

FIXES from v3.2.1 audit:
  - Too-long check now runs BEFORE short check (was shadowed)
  - Indic script detection uses ord() ranges instead of regex
    (regex pattern had encoding issues on some systems)
  - Legal signal check uses whole-word matching (word boundary)
  - Single-word error message now includes "too brief"
"""

import re
from config import QUERY_MIN_CHARS, QUERY_MAX_CHARS, QUERY_MIN_WORDS

# Legal signal words — at least one must appear as a standalone word
LEGAL_SIGNALS = [
    "ipc", "section", "accused", "court", "bail", "murder", "rape",
    "fraud", "appeal", "conviction", "acquittal", "sentence", "judge",
    "petitioner", "respondent", "plaintiff", "defendant", "fir", "charge",
    "arrest", "custody", "evidence", "witness", "verdict", "judgment",
    "crpc", "article", "writ", "habeas", "injunction", "decree",
    "theft", "robbery", "assault", "cheating", "dacoity", "offence",
    "offense", "criminal", "civil", "sessions", "magistrate", "high court",
    "supreme court", "tribunal", "acquit", "convict", "imprison",
    "sentenced", "charged", "alleged",
]


def _has_indic_script(text: str) -> bool:
    """
    Detect Indic script characters using Unicode code point ranges.
    Uses ord() checks — avoids regex encoding issues on all platforms.

    Ranges covered:
      0x0900–0x097F  Devanagari  (Hindi, Marathi, Sanskrit)
      0x0980–0x09FF  Bengali
      0x0A00–0x0A7F  Gurmukhi   (Punjabi)
      0x0A80–0x0AFF  Gujarati
      0x0B00–0x0B7F  Odia
      0x0B80–0x0BFF  Tamil
      0x0C00–0x0C7F  Telugu
      0x0C80–0x0CFF  Kannada
      0x0D00–0x0D7F  Malayalam
    """
    indic_count = 0
    for ch in text:
        cp = ord(ch)
        if (0x0900 <= cp <= 0x097F or   # Devanagari
            0x0980 <= cp <= 0x09FF or   # Bengali
            0x0A00 <= cp <= 0x0A7F or   # Gurmukhi
            0x0A80 <= cp <= 0x0AFF or   # Gujarati
            0x0B00 <= cp <= 0x0B7F or   # Odia
            0x0B80 <= cp <= 0x0BFF or   # Tamil
            0x0C00 <= cp <= 0x0C7F or   # Telugu
            0x0C80 <= cp <= 0x0CFF or   # Kannada
            0x0D00 <= cp <= 0x0D7F):    # Malayalam
            indic_count += 1
    return indic_count > len(text) * 0.25


def _has_legal_signal(text_lower: str) -> bool:
    """
    Check for at least one legal signal word.
    Uses word-boundary matching to avoid false positives from
    substrings (e.g. "like" inside "Unlike", "in" inside "injunction").
    """
    for signal in LEGAL_SIGNALS:
        # Use \b word boundary for single-word signals
        # Use plain 'in' check for multi-word signals like "high court"
        if " " in signal:
            if signal in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(signal) + r'\b', text_lower):
                return True
    return False


def validate_query(text: str) -> tuple:
    """
    Validate query before sending to NLP/ML pipeline.

    Returns:
        (True, "")                         — valid query
        (False, human-readable error msg)  — invalid query
    """
    # 1. Empty
    if not text or not text.strip():
        return False, (
            "Please describe your case. The search field is empty."
        )

    text = text.strip()

    # 2. Too long — check BEFORE word count to catch "word " * 1000
    if len(text) > QUERY_MAX_CHARS:
        return False, (
            f"Query too long ({len(text):,} characters, limit {QUERY_MAX_CHARS:,}). "
            f"Summarize the key charges, facts, and evidence in a few sentences. "
            f"For a full judgment text, use the PDF upload feature."
        )

    # 3. Too short (character count)
    if len(text) < QUERY_MIN_CHARS:
        return False, (
            f"Query too short ({len(text)} characters, minimum {QUERY_MIN_CHARS}). "
            f"Example: 'Accused charged under IPC Section 302 for murder "
            f"with eyewitness and forensic evidence.'"
        )

    # 4. Too few words
    word_count = len(text.split())
    if word_count < QUERY_MIN_WORDS:
        return False, (
            f"Query too brief ({word_count} word{'s' if word_count != 1 else ''}). "
            f"Please describe the charges, facts, and evidence in at least "
            f"{QUERY_MIN_WORDS} words."
        )

    # 5. Non-Latin / Indic script
    if _has_indic_script(text):
        return False, (
            "Query appears to be in a non-English script. "
            "LexAI's embedding model (LegalBERT) was trained on English legal text. "
            "Please enter your query in English for accurate results."
        )

    # 6. No legal signal
    text_lower = text.lower()
    if not _has_legal_signal(text_lower):
        return False, (
            "Query doesn't appear to describe a legal case. "
            "Please include legal context such as charges (IPC section), "
            "case type (murder, bail, fraud), court, or parties. "
            "Example: 'Accused charged under IPC 420 for cheating. Victim filed FIR.'"
        )

    return True, ""
