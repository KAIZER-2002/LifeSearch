from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


STOPWORDS = {
    "find", "show", "me", "the", "a", "an", "that", "which", "what", "did",
    "i", "work", "on", "about", "around", "in", "from", "where", "had",
    "remember", "my", "last", "this", "was", "for", "with", "to", "of",
    "at", "by", "or", "and", "is", "were", "been", "have", "has", "do",
    "does", "some", "any", "all", "so", "it", "its", "or"
}

FILE_EXTENSIONS = {
    "pdf", "docx", "doc", "txt", "md", "markdown", "png", "jpg", "jpeg",
    "py", "js", "ts", "go", "rs", "java", "cpp", "c", "h", "html", "css"
}

MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)

# Regex patterns for temporal extraction
TEMPORAL_PATTERNS = [
    re.compile(r"\baround\s+(" + MONTH_NAMES + r")\b", re.IGNORECASE),
    re.compile(r"\bin\s+(" + MONTH_NAMES + r")(\s+\d{4})?\b", re.IGNORECASE),
    re.compile(r"\bbetween\s+\d{1,2}(?:am|pm)?\s+and\s+\d{1,2}(?:am|pm)?(?:\s+on\s+\d{4}-\d{2}-\d{2})?\b", re.IGNORECASE),
    re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\blast\s+(week|month|year)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(week|month|year)\b", re.IGNORECASE),
    re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\btwo\s+days\s+ago\b", re.IGNORECASE),
]

ACTIVITY_PATTERN = re.compile(
    r"\bwhat\s+(?:did\s+i|was\s+i)\s+(?:work|do|working|doing)\b|\bwhat\s+files\s+did\s+i\b",
    re.IGNORECASE,
)

SCREENSHOT_PATTERN = re.compile(
    r"\bscreenshot\b|\bimage\s+of\b|\berror\s+screenshot\b|\bscreenshot\s+where\b|\bphoto\b|\bpicture\b",
    re.IGNORECASE,
)


@dataclass
class ParsedQuery:
    original: str
    terms: List[str] = field(default_factory=list)
    filename_hint: str = ""
    file_type: str = ""
    topic_hints: List[str] = field(default_factory=list)
    intent: str = "search"  # "search" | "activity" | "screenshot"
    time_expression: str = ""
    has_temporal: bool = False


class QueryParser:
    def parse(self, query: str) -> ParsedQuery:
        if not query or not query.strip():
            return ParsedQuery(original=query)

        raw = query.strip()
        
        # 1. Intent Detection
        if ACTIVITY_PATTERN.search(raw):
            intent = "activity"
        elif SCREENSHOT_PATTERN.search(raw):
            intent = "screenshot"
        else:
            intent = "search"

        # 2. Temporal Extraction
        time_expression = ""
        for pattern in TEMPORAL_PATTERNS:
            match = pattern.search(raw)
            if match:
                time_expression = match.group(0)
                break
        has_temporal = bool(time_expression)

        # 3. File Type & Filename Hint Extraction
        file_type = ""
        filename_hint = ""

        # Check for explicit filename e.g. "retrieval.py", "qdrant.pdf"
        filename_match = re.search(r"\b([a-zA-Z0-9_-]+\.(?:" + "|".join(FILE_EXTENSIONS) + r"))\b", raw, re.IGNORECASE)
        if filename_match:
            filename_hint = filename_match.group(1)
            ext = filename_hint.rsplit(".", 1)[-1].lower()
            file_type = ext

        if not file_type:
            words_lower = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9._-]+\b", raw)]
            if any(w in ("pdf", "docx", "doc", "txt", "md", "markdown") for w in words_lower):
                for w in words_lower:
                    if w in ("pdf", "docx", "doc", "txt", "md", "markdown"):
                        file_type = w
                        break
            elif any(w in ("screenshot", "image", "photo", "picture", "png", "jpg", "jpeg") for w in words_lower):
                file_type = "image"
            elif any(w in ("py", "js", "ts", "go", "rs", "java", "cpp", "c") for w in words_lower):
                file_type = "code"

        # 4. Extract terms & topics (cleaning out stopwords, time_expression, file_type tokens)
        clean_text = raw
        if time_expression:
            clean_text = re.sub(re.escape(time_expression), "", clean_text, flags=re.IGNORECASE)

        tokens = re.findall(r"\b[a-zA-Z0-9._-]+\b", clean_text)
        terms: List[str] = []
        topic_hints: List[str] = []

        for token in tokens:
            t_lower = token.lower()
            if t_lower in STOPWORDS:
                continue
            if t_lower in FILE_EXTENSIONS and len(t_lower) <= 4 and not filename_hint:
                continue
            if t_lower in ("screenshot", "image", "photo", "picture", "file", "files", "document", "documents"):
                continue
            if t_lower not in terms:
                terms.append(t_lower)
                topic_hints.append(t_lower)

        return ParsedQuery(
            original=raw,
            terms=terms,
            filename_hint=filename_hint,
            file_type=file_type,
            topic_hints=topic_hints,
            intent=intent,
            time_expression=time_expression,
            has_temporal=has_temporal,
        )
