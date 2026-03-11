from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    openalex_id: str
    title: str
    authors: list[str]
    abstract: str | None
    doi: str | None
    publication_date: str
    journal: str | None
    url: str
    relevance_score: float = 0.0
    matched_topic: str | None = None
    summary: str | None = None
    source: str = ""  # "topic", "author", "topic+author"

    @property
    def formatted_authors(self) -> str:
        if len(self.authors) <= 6:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:5]) + ", ... et al., " + self.authors[-1]


@dataclass
class Topic:
    label: str
    description: str


@dataclass
class TopicSection:
    label: str
    papers: list[Paper] = field(default_factory=list)


@dataclass
class Newsletter:
    topic_sections: list[TopicSection]
    author_papers: list[Paper]
    date_range: str
