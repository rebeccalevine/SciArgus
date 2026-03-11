from __future__ import annotations

import logging
from pathlib import Path

from .models import Topic

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def parse_topics(path: Path | None = None) -> list[Topic]:
    path = path or CONFIG_DIR / "topics.md"
    text = path.read_text(encoding="utf-8")
    topics: list[Topic] = []
    current_label: str | None = None
    current_desc_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("##"):
            if current_label is not None:
                desc = "\n".join(current_desc_lines).strip()
                topics.append(Topic(label=current_label, description=desc))
            current_label = line.lstrip("#").strip()
            current_desc_lines = []
        else:
            current_desc_lines.append(line)

    if current_label is not None:
        desc = "\n".join(current_desc_lines).strip()
        topics.append(Topic(label=current_label, description=desc))

    logger.info("Parsed %d topics", len(topics))
    return topics


def parse_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    logger.info("Parsed %d names from %s", len(names), path.name)
    return names


def parse_authors(path: Path | None = None) -> list[str]:
    return parse_names(path or CONFIG_DIR / "authors.md")


def parse_journals(path: Path | None = None) -> list[str]:
    return parse_names(path or CONFIG_DIR / "journals.md")
