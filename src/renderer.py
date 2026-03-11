from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .models import Newsletter

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def render(newsletter: Newsletter) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("newsletter.html")
    return template.render(newsletter=newsletter)
