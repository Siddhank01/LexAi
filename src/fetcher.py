import sqlite3, json, requests, time, logging
from pathlib import Path
from config import (
    INDIAN_KANOON_API_KEY, INDIAN_KANOON_BASE_URL,
    DB_PATH, START_WITH_N_CASES
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ── SCHEMA VERIFIED: SnehaDeshmukh/IndianBailJudgments-1200 ───────────────
# Fields: facts, judgment_reason, summary, ipc_sections, court, date,
#         bail_outcome, crime_type, case_title, judge, accused_name, etc.
# Text = combined facts + judgment_reason + summary (~870 chars avg)
DATASET_NAME = "SnehaDeshmukh/IndianBailJudgments-1200"
TEXT_FIELDS = ["facts", "judgment_reason", "summary"]  # combined into raw_text
COURT_FIELD = "court"
DATE_FIELD = "date"
VERDICT_FIELD = "bail_outcome"


def validate_schema(row: dict) -> tuple:
    """
    v3.1: Validate first row before processing all cases.
    Returns (is_valid: bool, reason: str)
    """
    for field in TEXT_FIELDS:
        if field not in row:
            return False, (
                f"Expected field '{field}' not found in row. "
                f"Available fields: {list(row.keys())}. "
                f"Dataset schema may have changed."
            )

    # Combine text fields and check length
    combined = " ".join(str(row.get(f, "")) for f in TEXT_FIELDS)
    if len(combined.strip()) < 50:
        return False, (
            f"Combined text fields too short ({len(combined)} chars). "
            f"Fields: {TEXT_FIELDS}"
        )
    return True, "ok"


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            court TEXT,
            date TEXT,
            raw_text TEXT,
            source TEXT,
            meta TEXT
        )
    """)
    conn.commit()
    return conn


def fetch_from_huggingface(max_cases=500):
    from datasets import load_dataset

    log.info(f"Loading {max_cases} diverse cases from {DATASET_NAME}...")
    ds = load_dataset(
        DATASET_NAME,
        split="train",
    )
    
    # Stratified sampling: shuffle to ensure diversity instead of just top N
    if len(ds) > max_cases:
        ds = ds.shuffle(seed=42).select(range(max_cases))
    else:
        log.warning(f"Dataset has {len(ds)} cases, requesting {max_cases}. Taking all.")
        ds = ds.shuffle(seed=42)

    if len(ds) == 0:
        raise ValueError("Dataset returned 0 rows.")

    first_row = dict(ds[0])
    valid, reason = validate_schema(first_row)
    if not valid:
        raise ValueError(
            f"SCHEMA VALIDATION FAILED: {reason}\n"
            f"Run the Phase 2.1 schema verification snippet first."
        )

    log.info(
        f"Schema valid. TEXT_FIELDS={TEXT_FIELDS}. "
        f"Processing {len(ds)} rows..."
    )

    empty_text_count = 0
    cases = []

    for i, row in enumerate(ds):
        # Combine text fields: facts + judgment_reason + summary
        text_parts = []
        for field in TEXT_FIELDS:
            val = str(row.get(field, "")).strip()
            if val and val.lower() != "none":
                text_parts.append(val)
        combined_text = " ".join(text_parts)

        if len(combined_text.strip()) < 50:
            empty_text_count += 1
            continue

        # Extract IPC sections from the dataset (already parsed!)
        ipc_raw = row.get("ipc_sections", "[]")
        try:
            if isinstance(ipc_raw, str):
                ipc_sections = json.loads(ipc_raw.replace("'", '"'))
            elif isinstance(ipc_raw, list):
                ipc_sections = ipc_raw
            else:
                ipc_sections = []
        except (json.JSONDecodeError, Exception):
            ipc_sections = []

        case_id = str(row.get("case_id", f"bail_{i}"))

        cases.append({
            "id": f"hf_{case_id}",
            "court": str(row.get(COURT_FIELD, "unknown")),
            "date": str(row.get(DATE_FIELD, "")),
            "raw_text": combined_text,
            "source": "huggingface",
            "meta": json.dumps({
                "length": len(combined_text),
                "row_index": i,
                "dataset": DATASET_NAME,
                "case_title": str(row.get("case_title", "")),
                "bail_outcome": str(row.get(VERDICT_FIELD, "")),
                "crime_type": str(row.get("crime_type", "")),
                "ipc_sections": ipc_sections,
                "judge": str(row.get("judge", "")),
                "accused_name": str(row.get("accused_name", "")),
                "bail_type": str(row.get("bail_type", "")),
            })
        })

    log.info(
        f"Loaded {len(cases)} valid cases. "
        f"{empty_text_count} skipped (empty text)."
    )

    if empty_text_count > max_cases * 0.3:
        log.warning(
            f"WARNING: {empty_text_count}/{max_cases} rows had empty text. "
            f"TEXT_FIELDS={TEXT_FIELDS} may be wrong."
        )

    return cases


def fetch_from_indian_kanoon(
    query: str,
    pages: int = 10,
    method: str = "GET"
) -> list:
    """
    Fetch cases from Indian Kanoon API.
    - Uses GET or POST based on what the API accepts
    - 1.5 second delay after every request
    - Retries up to 3 times on failure
    - Deduplicates by doc id or hash(title+court)
    """
    import time, hashlib

    if not INDIAN_KANOON_API_KEY:
        log.info("No Indian Kanoon API key. Skipping.")
        return []

    headers = {"Authorization": f"Token {INDIAN_KANOON_API_KEY}"}
    cases = []

    for page in range(1, pages + 1):
        # Retry logic — up to 3 attempts per page
        success = False
        for attempt in range(1, 4):
            try:
                if method == "GET":
                    r = requests.get(
                        f"{INDIAN_KANOON_BASE_URL}/search/",
                        params={"formInput": query, "pagenum": page},
                        headers=headers,
                        timeout=30
                    )
                else:
                    r = requests.post(
                        f"{INDIAN_KANOON_BASE_URL}/search/",
                        data={"formInput": query, "pagenum": page},
                        headers=headers,
                        timeout=30
                    )

                if r.status_code == 200:
                    docs = r.json().get("docs", [])
                    for doc in docs:
                        cases.append({
                            "id":       str(doc.get("tid", "")),
                            "court":    doc.get("court", "unknown"),
                            "date":     doc.get("publishdate", ""),
                            "raw_text": doc.get("doc", ""),
                            "source":   "indiankanoon",
                            "meta":     json.dumps(doc)
                        })
                    log.info(
                        f"Query '{query}' page {page}: "
                        f"{len(docs)} docs fetched"
                    )
                    success = True
                    break
                else:
                    log.warning(
                        f"Attempt {attempt}/3 — "
                        f"Status {r.status_code} for '{query}' page {page}"
                    )
            except Exception as e:
                log.warning(
                    f"Attempt {attempt}/3 — "
                    f"Error for '{query}' page {page}: {e}"
                )

            # Wait before retry
            time.sleep(2)

        if not success:
            log.error(
                f"Skipping '{query}' page {page} "
                f"after 3 failed attempts."
            )

        # Mandatory delay after every request (success or skip)
        time.sleep(1.5)

    log.info(f"Fetched {len(cases)} raw cases for '{query}'.")
    return cases


def deduplicate_cases(cases: list) -> list:
    """
    Remove duplicate cases using:
    1. doc id (tid) if available and non-empty
    2. hash(title + court) as fallback
    Returns unique cases only.
    """
    import hashlib
    seen = set()
    unique = []

    for c in cases:
        # Try id first
        case_id = c.get("id", "").strip()

        # Fallback: hash of raw_text first 200 chars + court
        if not case_id or case_id == "":
            raw = c.get("raw_text", "")[:200]
            court = c.get("court", "")
            case_id = hashlib.md5(
                f"{raw}{court}".encode()
            ).hexdigest()

        if case_id not in seen:
            seen.add(case_id)
            # Make sure the id field is set
            c["id"] = case_id
            unique.append(c)

    removed = len(cases) - len(unique)
    if removed > 0:
        log.info(f"Deduplication: removed {removed} duplicates.")
    return unique


def save_cases_to_db(cases, conn):
    inserted = 0
    for c in cases:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?)",
                (c["id"], c["court"], c["date"],
                 c["raw_text"], c["source"], c["meta"])
            )
            inserted += 1
        except Exception as e:
            log.error(f"Insert error {c['id']}: {e}")
    conn.commit()
    log.info(f"Saved {inserted} new cases to DB.")
    return inserted


def get_case_count(conn):
    return conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]


if __name__ == "__main__":
    import time

    conn = init_database()

    # Use whichever method worked in Step 1
    # Change "GET" to "POST" if Step 1 showed POST works
    API_METHOD = "POST"

    QUERIES = [
        ("IPC 302 murder conviction sessions court",     10),
        ("IPC 302 murder acquittal benefit of doubt",    10),
        ("IPC 376 rape conviction High Court",           10),
        ("IPC 420 fraud cheating conviction",            10),
        ("IPC 498A domestic violence matrimonial",       10),
        ("IPC 307 attempt murder conviction",            10),
        ("bail application murder rejected",             10),
        ("bail application granted Supreme Court",       10),
        ("appeal against conviction allowed High Court", 10),
        ("acquittal evidence insufficient witness",      10),
    ]

    TARGET_CASES = 5000
    all_new_cases = []

    for query, pages in QUERIES:
        current_count = get_case_count(conn)
        if current_count >= TARGET_CASES:
            print(f"Target {TARGET_CASES} reached. Stopping.")
            break

        print(f"\nFetching: '{query}' ({pages} pages)...")
        raw_cases = fetch_from_indian_kanoon(
            query, pages=pages, method=API_METHOD
        )
        all_new_cases.extend(raw_cases)
        print(f"  Got {len(raw_cases)} raw cases")

    # Deduplicate everything before saving
    print(f"\nTotal raw cases fetched: {len(all_new_cases)}")
    unique_cases = deduplicate_cases(all_new_cases)
    print(f"After deduplication: {len(unique_cases)}")

    saved = save_cases_to_db(unique_cases, conn)
    final_count = get_case_count(conn)

    print(f"\nNew cases saved: {saved}")
    print(f"Total cases in DB: {final_count}")
    conn.close()

    if final_count < TARGET_CASES:
        print(
            f"\nNote: Got {final_count} cases (target {TARGET_CASES})."
            f"\nRun fetcher again with different queries to add more."
        )
    else:
        print(f"\nTarget reached: {final_count} cases in DB.")
