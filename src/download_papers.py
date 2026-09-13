"""
Download research papers on PEFT / LLM fine-tuning techniques from arxiv.

Each paper is saved to data/papers/ with a descriptive filename.
Run this script once to populate the corpus:

    python -m src.download_papers
"""

import logging
import time
import urllib.request
from pathlib import Path

from src.config import PAPERS_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paper catalogue
# ──────────────────────────────────────────────
PAPERS: list[dict[str, str]] = [
    {
        "arxiv_id": "2106.09685",
        "filename": "lora.pdf",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "year": "2021",
    },
    {
        "arxiv_id": "2305.14314",
        "filename": "qlora.pdf",
        "title": "QLoRA: Efficient Finetuning of Quantized Language Models",
        "year": "2023",
    },
    {
        "arxiv_id": "2101.00190",
        "filename": "prefix_tuning.pdf",
        "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
        "year": "2021",
    },
    {
        "arxiv_id": "1902.00751",
        "filename": "adapters_houlsby.pdf",
        "title": "Parameter-Efficient Transfer Learning for NLP",
        "year": "2019",
    },
    {
        "arxiv_id": "2103.10385",
        "filename": "p_tuning.pdf",
        "title": "GPT Understands, Too (P-Tuning)",
        "year": "2021",
    },
    {
        "arxiv_id": "2110.07602",
        "filename": "p_tuning_v2.pdf",
        "title": "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning",
        "year": "2021",
    },
    {
        "arxiv_id": "2205.05638",
        "filename": "ia3_t_few.pdf",
        "title": "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning",
        "year": "2022",
    },
    {
        "arxiv_id": "2104.08691",
        "filename": "prompt_tuning_lester.pdf",
        "title": "The Power of Scale for Parameter-Efficient Prompt Tuning",
        "year": "2021",
    },
]


def download_paper(paper: dict[str, str], dest_dir: Path) -> Path:
    """Download a single paper PDF from arxiv."""
    dest_path = dest_dir / paper["filename"]

    if dest_path.exists():
        logger.info(f"  ✓ Already exists: {paper['filename']}")
        return dest_path

    url = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    logger.info(f"  ↓ Downloading: {paper['title']}")
    logger.info(f"    URL: {url}")

    # arxiv prefers a User-Agent header; add a small delay to be polite.
    # BUGFIX: the previous version built this Request object and then called
    # urlretrieve(url, ...), which ignores it entirely and sends the default
    # Python-urllib UA — arxiv sometimes rate-limits that. Use urlopen so the
    # header is actually sent.
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RAG-Portfolio-Project/1.0 (student project)"},
    )
    with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Saved: {paper['filename']} ({size_mb:.1f} MB)")
    return dest_path


def download_all_papers() -> list[Path]:
    """Download all papers in the catalogue to PAPERS_DIR."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {len(PAPERS)} papers to {PAPERS_DIR}")
    logger.info("=" * 60)

    downloaded: list[Path] = []
    for i, paper in enumerate(PAPERS, 1):
        logger.info(f"\n[{i}/{len(PAPERS)}] {paper['title']} ({paper['year']})")
        path = download_paper(paper, PAPERS_DIR)
        downloaded.append(path)
        # Be polite to arxiv — wait 3 seconds between downloads
        if i < len(PAPERS):
            time.sleep(3)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done! {len(downloaded)} papers in {PAPERS_DIR}")
    return downloaded


def list_papers() -> None:
    """Print the paper catalogue as a formatted table."""
    print(f"\n{'#':<4} {'Method':<20} {'ArXiv ID':<14} {'Year':<6} {'Filename'}")
    print("-" * 80)
    for i, p in enumerate(PAPERS, 1):
        print(f"{i:<4} {p['filename']:<20} {p['arxiv_id']:<14} {p['year']:<6} {p['title']}")


# ──────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("\nPEFT Research Papers - Download Script")
    print("=" * 60)
    list_papers()
    print()

    paths = download_all_papers()

    print("\nFiles in data/papers/:")
    for p in sorted(PAPERS_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<30} {size_mb:.1f} MB")
