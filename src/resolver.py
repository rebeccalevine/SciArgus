from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "resolved_ids.json"
CACHE_TTL_DAYS = 30
OPENALEX_BASE = "https://api.openalex.org"
MAILTO = "sciargus.grm@gmail.com"
POLITE_DELAY = 0.5


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        age_days = (time.time() - data.get("timestamp", 0)) / 86400
        if age_days < CACHE_TTL_DAYS:
            logger.info("Cache is %.1f days old, using cached IDs", age_days)
            return data
        logger.info("Cache expired (%.1f days old), re-resolving", age_days)
    return {}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["timestamp"] = time.time()
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_one(endpoint: str, name: str, session: requests.Session) -> str | None:
    params = {
        "filter": f"display_name.search:{name}",
        "per_page": "1",
        "select": "id,display_name",
        "mailto": MAILTO,
    }
    if endpoint == "authors":
        params["sort"] = "cited_by_count:desc"

    try:
        resp = session.get(f"{OPENALEX_BASE}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            resolved_id = results[0]["id"]
            resolved_name = results[0]["display_name"]
            logger.info("  %s -> %s (%s)", name, resolved_name, resolved_id)
            return resolved_id
        logger.warning("  No results for %s", name)
        return None
    except requests.RequestException as e:
        logger.warning("  Failed to resolve %s: %s", name, e)
        return None


def resolve_all(
    author_names: list[str], journal_names: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve author and journal names to OpenAlex IDs.

    Returns (author_map, journal_map) where keys are names and values are IDs.
    """
    cache = _load_cache()
    if cache.get("authors") and cache.get("journals"):
        author_map = {k: v for k, v in cache["authors"].items() if v}
        journal_map = {k: v for k, v in cache["journals"].items() if v}
        logger.info(
            "Using cached: %d authors, %d journals", len(author_map), len(journal_map)
        )
        return author_map, journal_map

    session = requests.Session()
    author_map: dict[str, str] = {}
    journal_map: dict[str, str] = {}

    logger.info("Resolving %d author names...", len(author_names))
    for name in author_names:
        resolved = _resolve_one("authors", name, session)
        if resolved:
            author_map[name] = resolved
        time.sleep(POLITE_DELAY)

    logger.info("Resolving %d journal names...", len(journal_names))
    for name in journal_names:
        resolved = _resolve_one("sources", name, session)
        if resolved:
            journal_map[name] = resolved
        time.sleep(POLITE_DELAY)

    _save_cache(
        {
            "authors": {
                name: author_map.get(name) for name in author_names
            },
            "journals": {
                name: journal_map.get(name) for name in journal_names
            },
        }
    )

    logger.info(
        "Resolved %d/%d authors, %d/%d journals",
        len(author_map),
        len(author_names),
        len(journal_map),
        len(journal_names),
    )
    return author_map, journal_map
