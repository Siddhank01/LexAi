"""
Fetch full judgment text for Indian Kanoon cases
that currently have short snippet text only.
Calls /doc/{tid}/ endpoint for each case.
Saves progress every 100 cases so it can resume
if interrupted.
"""

import sqlite3, json, time, requests, logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

DB_PATH        = "data/judgments.db"
API_KEY        = os.getenv("INDIAN_KANOON_API_KEY", "")
BASE_URL       = "https://api.indiankanoon.org"
PROGRESS_FILE  = "data/raw/fetch_progress.json"
MIN_TEXT_LEN   = 500   # anything shorter is a snippet, not full text


def load_progress() -> set:
    """Load set of already-fetched case IDs."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(fetched_ids: set):
    """Save progress so we can resume if interrupted."""
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(fetched_ids), f)


def fetch_full_text(tid: str, attempt: int = 1) -> str:
    """
    Fetch full judgment text for a single case.
    Retries up to 3 times on failure.
    Returns empty string if all attempts fail.
    """
    if not tid or tid.strip() == "":
        return ""

    headers = {"Authorization": f"Token {API_KEY}"}
    url = f"{BASE_URL}/doc/{tid}/"

    for i in range(1, 4):
        try:
            r = requests.post(url, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                text = data.get("doc", "")
                if text and len(text) > MIN_TEXT_LEN:
                    return text
                else:
                    return ""
            elif r.status_code == 429:
                # Rate limited — wait longer
                log.warning(f"Rate limited on {tid}. Waiting 10s...")
                time.sleep(10)
            else:
                log.warning(
                    f"Attempt {i}/3 — Status {r.status_code} for tid={tid}"
                )
        except Exception as e:
            log.warning(f"Attempt {i}/3 — Error for tid={tid}: {e}")

        time.sleep(2)

    return ""


def get_short_text_cases(conn) -> list:
    """
    Get all Indian Kanoon cases with short text (snippets only).
    These need full text fetched.
    """
    rows = conn.execute(
        """
        SELECT id, court, date, raw_text, source, meta
        FROM cases
        WHERE source = 'indiankanoon'
        AND LENGTH(raw_text) < ?
        """,
        (MIN_TEXT_LEN,)
    ).fetchall()
    return rows


def update_case_text(conn, case_id: str, full_text: str):
    """Update a case's raw_text with the full judgment."""
    conn.execute(
        "UPDATE cases SET raw_text = ? WHERE id = ?",
        (full_text, case_id)
    )


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: INDIAN_KANOON_API_KEY not found in .env")
        exit(1)

    conn = sqlite3.connect(DB_PATH)
    
    # Get cases that need full text
    short_cases = get_short_text_cases(conn)
    print(f"Cases needing full text: {len(short_cases)}")
    
    if len(short_cases) == 0:
        print("All cases already have full text. Nothing to do.")
        conn.close()
        exit(0)

    # Load progress (resume support)
    fetched_ids = load_progress()
    remaining = [r for r in short_cases if r[0] not in fetched_ids]
    print(f"Already fetched: {len(fetched_ids)}")
    print(f"Remaining: {len(remaining)}")
    
    estimated_minutes = len(remaining) * 1.5 / 60
    print(f"Estimated time: {estimated_minutes:.0f} minutes")
    print(f"Starting in 3 seconds... (Ctrl+C to stop, progress is saved)")
    time.sleep(3)

    success_count = 0
    skip_count    = 0
    batch_size    = 100

    for i, row in enumerate(remaining):
        case_id = row[0]
        meta    = {}
        try:
            meta = json.loads(row[5]) if row[5] else {}
        except Exception:
            pass

        # Get tid from meta or from id
        tid = str(meta.get("tid", "")).strip()
        if not tid:
            # Try extracting from case_id
            tid = case_id.replace("hf_", "").replace("pq_", "")

        # Progress display every 50 cases
        if i % 50 == 0:
            print(
                f"  [{i}/{len(remaining)}] "
                f"fetched={success_count} "
                f"skipped={skip_count} "
                f"tid={tid[:10]}..."
            )

        # Fetch full text
        full_text = fetch_full_text(tid)

        if full_text:
            update_case_text(conn, case_id, full_text)
            fetched_ids.add(case_id)
            success_count += 1
        else:
            skip_count += 1

        # Save progress and commit every 100 cases
        if (i + 1) % batch_size == 0:
            conn.commit()
            save_progress(fetched_ids)
            log.info(
                f"Progress saved. "
                f"Batch {(i+1)//batch_size} done. "
                f"Success: {success_count} Skip: {skip_count}"
            )

        # Delay between requests
        time.sleep(1.5)

    # Final commit and save
    conn.commit()
    save_progress(fetched_ids)

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"Full text fetched: {success_count}")
    print(f"Skipped (no text): {skip_count}")
    print(f"{'='*50}")
    print(f"\nNext step: run python src/nlp_pipeline.py")

    conn.close()
