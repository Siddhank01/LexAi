"""
LexAI — HuggingFace data downloader.

Downloads large ML artefacts from HuggingFace Hub at startup.
Files are cached locally — subsequent runs skip the download.

Dataset repo: https://huggingface.co/datasets/Satyam810/lexai-data
"""

import os
import sys
import hashlib
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────

HF_DATASET_REPO = "Satyam810/lexai-data"   # ← your HF dataset repo name

# Files to download: { local_path: filename_in_hf_repo }
DATA_FILES = {
    "data/processed/cases.json":         "cases.json",
    "data/processed/faiss.index":        "faiss.index",
    "data/processed/embeddings.npy":     "embeddings.npy",
    "data/processed/cluster_labels.npy": "cluster_labels.npy",
    "data/processed/coords_2d.npy":      "coords_2d.npy",
}


# ── Downloader ────────────────────────────────────────────────────────────────

def download_if_missing(verbose: bool = True) -> bool:
    """
    Download missing data files from HuggingFace Hub.

    Returns True if all files are present after download, False on error.
    Call this at the TOP of app.py before any other imports that need the data.
    """

    missing = [
        (local, hf_name)
        for local, hf_name in DATA_FILES.items()
        if not Path(local).exists()
    ]

    if not missing:
        if verbose:
            print("✅ All data files present — skipping download.")
        return True

    if verbose:
        print(f"📥 Downloading {len(missing)} missing file(s) from HuggingFace...")
        for local, _ in missing:
            print(f"   • {local}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "❌ huggingface_hub not installed. "
            "Add it to requirements.txt and re-deploy."
        )
        return False

    # Ensure directories exist
    for local, _ in missing:
        Path(local).parent.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for local, hf_name in missing:
        try:
            if verbose:
                print(f"   ⬇️  Downloading {hf_name}...")

            downloaded = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=hf_name,
                repo_type="dataset",
                local_dir="data/processed",
                local_dir_use_symlinks=False,
            )

            # Verify file landed in the right place
            if Path(downloaded).exists():
                size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
                if verbose:
                    print(f"   ✅ {hf_name} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {hf_name} — download succeeded but file not found at {downloaded}")
                all_ok = False

        except Exception as e:
            print(f"   ❌ Failed to download {hf_name}: {e}")
            all_ok = False

    if all_ok and verbose:
        print("✅ All data files ready.")
    elif not all_ok and verbose:
        print(
            "⚠️  Some files failed to download. "
            "Check your HF_DATASET_REPO setting and that the dataset is public."
        )

    return all_ok


if __name__ == "__main__":
    # Can be run standalone: python download_data.py
    success = download_if_missing(verbose=True)
    sys.exit(0 if success else 1)
