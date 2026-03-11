from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

from google import genai

from .config_parser import parse_authors, parse_journals, parse_topics
from .mailer import send_email
from .models import Newsletter, TopicSection
from .renderer import render
from .resolver import resolve_all
from .scraper import scrape_author_papers, scrape_topic_papers
from .summarizer import RateLimiter, generate_summaries, score_papers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _date_range_label() -> str:
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    return f"{week_ago.strftime('%b %d')} \u2013 {today.strftime('%b %d, %Y')}"


def _select_topic_papers(
    papers: list, topics: list, max_total: int = 10
) -> list[TopicSection]:
    """Select up to max_total papers distributed across topics."""
    # Group papers by matched_topic
    by_topic: dict[str, list] = {t.label: [] for t in topics}
    for p in papers:
        if p.matched_topic and p.matched_topic in by_topic:
            by_topic[p.matched_topic].append(p)

    # Sort each group by score descending
    for group in by_topic.values():
        group.sort(key=lambda p: p.relevance_score, reverse=True)

    # Round-robin: 1 per topic first, then fill by score
    selected_ids: set[str] = set()
    sections: dict[str, list] = {t.label: [] for t in topics}
    count = 0

    # Round 1: top paper from each topic
    for t in topics:
        if count >= max_total:
            break
        if by_topic[t.label]:
            paper = by_topic[t.label][0]
            sections[t.label].append(paper)
            selected_ids.add(paper.openalex_id)
            count += 1

    # Round 2: fill remaining slots by score across all topics
    if count < max_total:
        remaining = []
        for t in topics:
            for p in by_topic[t.label]:
                if p.openalex_id not in selected_ids:
                    remaining.append(p)
        remaining.sort(key=lambda p: p.relevance_score, reverse=True)
        for p in remaining:
            if count >= max_total:
                break
            if p.matched_topic:
                sections[p.matched_topic].append(p)
                selected_ids.add(p.openalex_id)
                count += 1

    # Build TopicSection list, omitting empty topics
    result = []
    for t in topics:
        if sections[t.label]:
            result.append(TopicSection(label=t.label, papers=sections[t.label]))

    return result


def main() -> None:
    gemini_key = os.environ.get("GEMINI_KEY")
    app_password = os.environ.get("GOOGLE_APP_PASSWORD")
    sender_email = os.environ.get("SENDER_EMAIL")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not all([gemini_key, app_password, sender_email, receiver_email]):
        logger.error(
            "Missing environment variables. Need: GEMINI_KEY, GOOGLE_APP_PASSWORD, "
            "SENDER_EMAIL, RECEIVER_EMAIL"
        )
        sys.exit(1)

    # 1. Parse config
    topics = parse_topics()
    author_names = parse_authors()
    journal_names = parse_journals()

    # 2. Resolve names -> OpenAlex IDs
    author_map, journal_map = resolve_all(author_names, journal_names)
    author_ids = list(author_map.values())
    journal_ids = list(journal_map.values())

    if not journal_ids:
        logger.error("No journal IDs resolved, cannot search for papers")
        sys.exit(1)

    # 3. Scrape papers
    topic_papers = scrape_topic_papers(topics, journal_ids)
    author_papers = scrape_author_papers(author_ids) if author_ids else {}

    # 4. Deduplicate: papers in both pools -> topic pool wins
    for oa_id, paper in author_papers.items():
        if oa_id in topic_papers:
            topic_papers[oa_id].source = "topic+author"
        else:
            author_papers[oa_id] = paper

    # Remove duplicates from author pool
    author_only = {
        oa_id: p for oa_id, p in author_papers.items() if oa_id not in topic_papers
    }

    all_papers = list(topic_papers.values()) + list(author_only.values())

    if not all_papers:
        logger.info("No papers found this week. No email will be sent.")
        return

    logger.info("Total unique papers: %d", len(all_papers))

    # 5. Phase A: LLM relevance scoring
    client = genai.Client(api_key=gemini_key)
    limiter = RateLimiter()
    score_papers(client, all_papers, topics, limiter)

    # 6. Select top papers from each pool
    topic_pool = [p for p in all_papers if p.source in ("topic", "topic+author")]
    author_pool = [p for p in all_papers if p.source == "author"]

    topic_sections = _select_topic_papers(topic_pool, topics, max_total=10)
    selected_topic_ids = set()
    for section in topic_sections:
        for p in section.papers:
            selected_topic_ids.add(p.openalex_id)

    # Author pool: exclude papers already in topic sections, sort by score
    author_candidates = [
        p for p in author_pool if p.openalex_id not in selected_topic_ids
    ]
    author_candidates.sort(key=lambda p: p.relevance_score, reverse=True)
    selected_author = author_candidates[:10]

    # 7. Phase B: LLM summaries for selected papers
    all_selected = []
    for section in topic_sections:
        all_selected.extend(section.papers)
    all_selected.extend(selected_author)

    generate_summaries(client, all_selected, topics, limiter)

    # 8. Build newsletter
    date_range = _date_range_label()
    newsletter = Newsletter(
        topic_sections=topic_sections,
        author_papers=selected_author,
        date_range=date_range,
    )

    # 9. Render HTML
    html = render(newsletter)

    # 10. Send email
    subject = f"SciArgus Weekly \u2014 {date_range}"
    send_email(html, subject, sender_email, app_password, receiver_email)
    logger.info("Newsletter sent successfully!")


if __name__ == "__main__":
    main()
