"""Load and chunk markdown business docs for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "docs"


@dataclass
class Chunk:
    id: str
    title: str
    source: str
    text: str


def _split_paragraphs(text: str, max_chars: int = 700) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= max_chars:
                buf = para
            else:
                # hard wrap long paragraphs
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").title()
        for i, part in enumerate(_split_paragraphs(raw)):
            chunks.append(
                Chunk(
                    id=f"{path.stem}-{i}",
                    title=title,
                    source=path.name,
                    text=part,
                )
            )
    return chunks