from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import requests

from .models import Paper, Topic

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
MAILTO = "sciargus.grm@gmail.com"
SELECT_FIELDS = "id,title,authorships,primary_location,publication_date,abstract_inverted_index,doi"
POLITE_DELAY = 0.5
MAX_JOURNAL_IDS_PER_BATCH = 50


def _date_range() -> tuple[str, str]:
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    return str(week_ago), str(today)


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _parse_paper(work: dict, source: str) -> Paper:
    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name")
        if name:
            authors.append(name)

    location = work.get("primary_location") or {}
    source_info = location.get("source") or {}
    journal = source_info.get("display_name")

    doi = work.get("doi")
    url = doi if doi else work.get("id", "")

    return Paper(
        openalex_id=work["id"],
        title=work.get("title", "Untitled"),
        authors=authors,
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        doi=doi,
        publication_date=work.get("publication_date", ""),
        journal=journal,
        url=url,
        source=source,
    )


def _fetch_works(
    session: requests.Session, params: dict, source_tag: str
) -> list[Paper]:
    params["mailto"] = MAILTO
    params["select"] = SELECT_FIELDS
    papers = []
    try:
        resp = session.get(f"{OPENALEX_BASE}/works", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for work in data.get("results", []):
            papers.append(_parse_paper(work, source_tag))
        logger.info("  Fetched %d papers", len(papers))
    except requests.RequestException as e:
        logger.warning("  OpenAlex query failed: %s", e)
    return papers


def scrape_topic_papers(
    topics: list[Topic],
    journal_ids: list[str],
    session: requests.Session | None = None,
) -> dict[str, Paper]:
    """Search for papers matching each topic, filtered by tracked journals."""
    session = session or requests.Session()
    from_date, to_date = _date_range()
    papers: dict[str, Paper] = {}

    # Split journal IDs into batches of MAX_JOURNAL_IDS_PER_BATCH
    journal_batches = [
        journal_ids[i : i + MAX_JOURNAL_IDS_PER_BATCH]
        for i in range(0, len(journal_ids), MAX_JOURNAL_IDS_PER_BATCH)
    ]

    for topic in topics:
        logger.info("Searching topic: %s", topic.label)
        for batch in journal_batches:
            source_filter = "|".join(batch)
            params = {
                "search": topic.label,
                "filter": (
                    f"primary_location.source.id:{source_filter},"
                    f"from_publication_date:{from_date},"
                    f"to_publication_date:{to_date}"
                ),
                "per_page": "25",
            }
            for paper in _fetch_works(session, params, "topic"):
                if paper.openalex_id not in papers:
                    papers[paper.openalex_id] = paper
            time.sleep(POLITE_DELAY)

    logger.info("Topic search: %d unique papers", len(papers))
    return papers


def scrape_author_papers(
    author_ids: list[str],
    session: requests.Session | None = None,
) -> dict[str, Paper]:
    """Fetch recent papers by tracked authors."""
    session = session or requests.Session()
    from_date, to_date = _date_range()
    papers: dict[str, Paper] = {}

    # All author IDs in batches (limit ~50 per OR filter)
    author_batches = [
        author_ids[i : i + MAX_JOURNAL_IDS_PER_BATCH]
        for i in range(0, len(author_ids), MAX_JOURNAL_IDS_PER_BATCH)
    ]

    logger.info("Searching %d tracked authors...", len(author_ids))
    for batch in author_batches:
        author_filter = "|".join(batch)
        params = {
            "filter": (
                f"authorships.author.id:{author_filter},"
                f"from_publication_date:{from_date},"
                f"to_publication_date:{to_date}"
            ),
            "per_page": "100",
        }
        for paper in _fetch_works(session, params, "author"):
            if paper.openalex_id not in papers:
                papers[paper.openalex_id] = paper
        time.sleep(POLITE_DELAY)

    logger.info("Author search: %d unique papers", len(papers))
    return papers
