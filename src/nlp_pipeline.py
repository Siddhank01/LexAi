import re, json, sqlite3, spacy
from pathlib import Path
from config import DB_PATH, CASES_JSON_PATH, SPACY_MODEL

nlp = spacy.load(SPACY_MODEL)

VERDICT_PATTERNS = {
    "acquitted": [
        r"\b(acquitt|not\s+guilty|discharg)\w*\b",
        r"\bset\s+aside.*convict\w*\b",
        r"\bconviction\b.*\bset\s+aside\b",
        r"\border\s+of\s+acquittal\b.*\bconfirm\w*",
        r"\bdirected\s+to\s+be\s+released\b",
        r"\breleased\s+on\s+probation\b",
        r"\bwarrant\s+of\s+conviction\b.*\bquash\w*",
        r"\bbenefit\s+of\s+(the\s+)?doubt\b",
    ],
    "convicted": [
        r"\b(convict|found\s+guilty|sentenced\s+to|imprisonment\s+for)\w*\b",
        r"\bplea\s+of\s+guilty\b.*\baccepted?\b",
        r"\bwarrant\s+of\s+commitment\b",
        r"\bsentence\s+(is\s+)?maintained\b",
    ],
    "appeal_allowed": [
        r"\bappeal\s+(is\s+)?(hereby\s+)?allow\w*\b",
        r"\bappeal\s+(is\s+)?(hereby\s+)?succeed\w*\b",
    ],
    "appeal_dismissed": [
        r"\bappeal\s+(is\s+)?(hereby\s+)?dismiss\w*\b",
        r"\bappeal\s+(is\s+)?(hereby\s+)?reject\w*\b",
        r"\bappeal\s+(is\s+)?(hereby\s+)?fail\w*\b",
    ],
    "bail_granted": [
        r"\bbail\s+(is\s+)?grant\w*\b",
        r"\banticipatory\s+bail\b.*\ballow\w*\b",
        r"\bbail\s+application\b.*\ballow\w*\b",
        r"\benlarged\s+on\s+bail\b",
        r"\breleased\s+on\s+bail\b",
        r"\bbail\s+pray\w*\b.*\ballow\w*\b",
    ],
    "bail_rejected": [
        r"\b(bail\s+(is\s+)?reject\w*|refused\s+bail)\b",
        r"\bbail\s+application\b.*\bdismiss\w*\b",
        r"\bbail\s+application\b.*\breject\w*\b",
        r"\bremanded\s+(to\s+)?(judicial\s+)?custody\b",
        r"\bbail\s+application\b.*\brefus\w*\b",
        r"\bbail\s+pray\w*\b.*\b(dismiss|reject|refus)\w*\b",
        r"\bbail\s+(is\s+)?denied\b",
    ],
    "sentence_modified": [
        r"\bsentence\s+(is\s+)?reduc\w*\b",
        r"\bsentence\s+(is\s+)?enhanc\w*\b",
        r"\bperiod\s+already\s+undergone\b",
        r"\bsentence\s+commut\w*\b",
    ],
}

IPC_PATTERN = r"""
    (?:section|sec\.?|u/?s\.?)\s*
    (\d{1,3}[A-Za-z]?)
    (?:\s*/\s*(\d{1,3}[A-Za-z]?))*
    \s*(?:of\s+the\s+)?
    (?:Indian\s+Penal\s+Code|IPC|CrPC|Cr\.P\.C\.)?
"""

CASE_TYPE_KEYWORDS = {
    "criminal":       ["ipc", "crpc", "murder", "rape", "theft",
                       "robbery", "fraud", "cheating", "assault", "dacoity"],
    "civil":          ["contract", "property", "partition", "injunction",
                       "damages", "specific performance", "tort"],
    "constitutional": ["article 14", "article 19", "article 21",
                       "fundamental right", "writ", "habeas corpus"],
    "family":         ["divorce", "maintenance", "custody", "adoption",
                       "matrimonial", "hindu marriage"],
    "labour":         ["workman", "dismissal", "retrenchment", "labour court",
                       "industrial dispute", "provident fund"],
}

EVIDENCE_KEYWORDS = [
    "forensic", "dna", "fingerprint", "eyewitness",
    "confession", "cctv", "post mortem", "ballistic",
    "circumstantial", "documentary"
]


def clean_text(text):
    if not text: return ""
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def extract_verdict(text):
    text_lower = text.lower()
    for label, patterns in VERDICT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.VERBOSE):
                return label
    return "unknown"


def extract_ipc_sections(text):
    matches = re.findall(IPC_PATTERN, text, re.IGNORECASE | re.VERBOSE)
    sections = []
    for m in matches:
        if isinstance(m, tuple):
            sections.extend([x for x in m if x])
        else:
            sections.append(m)
    return sorted(set(s.upper() for s in sections if s))


def extract_case_type(text):
    text_lower = text.lower()
    scores = {
        k: sum(text_lower.count(kw) for kw in v)
        for k, v in CASE_TYPE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def extract_entities(text):
    doc = nlp(text[:5000])
    ents = {"persons": [], "organizations": [], "locations": [], "dates": []}
    for e in doc.ents:
        if e.label_ == "PERSON":      ents["persons"].append(e.text)
        elif e.label_ == "ORG":       ents["organizations"].append(e.text)
        elif e.label_ in ("GPE","LOC"): ents["locations"].append(e.text)
        elif e.label_ == "DATE":      ents["dates"].append(e.text)
    return {k: list(set(v)) for k, v in ents.items()}


def extract_evidence_types(text):
    text_lower = text.lower()
    return [kw for kw in EVIDENCE_KEYWORDS if kw in text_lower]


def process_case(row):
    case_id, court, date, raw_text, source, meta_str = row
    clean = clean_text(raw_text)

    # Parse stored metadata from fetcher (contains pre-extracted fields)
    try:
        meta = json.loads(meta_str) if meta_str else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}

    # Extract verdict from text via regex
    verdict_from_text = extract_verdict(clean)

    # Also use pre-extracted bail_outcome from dataset metadata as fallback
    bail_outcome = meta.get("bail_outcome", "").strip().lower()
    if verdict_from_text == "unknown" and bail_outcome:
        if bail_outcome in ("granted", "grant"):
            verdict_from_text = "bail_granted"
        elif bail_outcome in ("rejected", "reject", "denied"):
            verdict_from_text = "bail_rejected"
        elif bail_outcome in ("partly granted", "partial"):
            verdict_from_text = "bail_granted"

    # Extract IPC sections from text
    ipc_from_text = extract_ipc_sections(clean)

    # Merge with pre-extracted IPC from dataset metadata
    ipc_from_meta = meta.get("ipc_sections", [])
    if isinstance(ipc_from_meta, str):
        try:
            ipc_from_meta = json.loads(ipc_from_meta.replace("'", '"'))
        except Exception:
            ipc_from_meta = []
    # Normalize: ensure all are uppercase strings
    ipc_from_meta = [str(s).upper() for s in ipc_from_meta if s]
    merged_ipc = sorted(set(ipc_from_text + ipc_from_meta))

    return {
        "id": case_id,
        "court": court or "unknown",
        "date": date or "",
        "source": source or "unknown",
        "text": clean,
        "verdict": verdict_from_text,
        "ipc_sections": merged_ipc,
        "case_type": extract_case_type(clean),
        "entities": extract_entities(clean),
        "evidence_types": extract_evidence_types(clean),
        "text_length": len(clean),
        "crime_type": meta.get("crime_type", ""),
        "case_title": meta.get("case_title", ""),
    }


def run_pipeline(limit=None):
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT id, court, date, raw_text, source, meta FROM cases"
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    conn.close()

    print(f"Processing {len(rows)} cases...")
    processed = []
    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  [{i}/{len(rows)}]")
        try:
            processed.append(process_case(row))
        except Exception as e:
            print(f"  Error row {i}: {e}")

    Path(CASES_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    verdicts = {}
    for c in processed:
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1

    total = len(processed)
    unknown_pct = verdicts.get("unknown", 0) / total * 100

    print(f"\n{'='*55}")
    print(f"VERDICT DISTRIBUTION ({total} cases):")
    for v, count in sorted(verdicts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {v:<22} {count:4d}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*55}")

    # IPC coverage stats
    has_ipc = sum(1 for c in processed if c["ipc_sections"])
    print(f"\nIPC section coverage: {has_ipc}/{total} ({has_ipc/total*100:.0f}%)")

    # Case type distribution
    case_types = {}
    for c in processed:
        case_types[c["case_type"]] = case_types.get(c["case_type"], 0) + 1
    print(f"\nCASE TYPE DISTRIBUTION:")
    for ct, count in sorted(case_types.items(), key=lambda x: -x[1]):
        print(f"  {ct:<22} {count:4d}")

    # Evidence coverage
    has_evidence = sum(1 for c in processed if c["evidence_types"])
    print(f"\nEvidence coverage: {has_evidence}/{total} ({has_evidence/total*100:.0f}%)")

    if unknown_pct > 60:
        print(f"\nHARD STOP: {unknown_pct:.0f}% unknown verdicts.")
        print("DEBUG STEPS:")
        print("1. Check raw text samples from DB")
        print("2. Add missing regex patterns to VERDICT_PATTERNS")
        print("3. Re-run nlp_pipeline.py until unknown% < 40%")
    elif unknown_pct > 40:
        print(f"\nWARNING: {unknown_pct:.0f}% unknown. Proceed to Phase 4 with caution.")
    else:
        print(f"\nVerdict coverage good ({100-unknown_pct:.0f}% classified). Proceed.")

    return processed


if __name__ == "__main__":
    run_pipeline()
