from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .query_parser import ParsedQuery
from .temporal import TimeRange

WEIGHT_FTS = 0.35
WEIGHT_SEMANTIC = 0.25
WEIGHT_FILENAME = 0.15
WEIGHT_TEMPORAL = 0.12
WEIGHT_EPISODE = 0.08
WEIGHT_MEMORY = 0.05


def parse_iso_utc(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def normalize_fts_ranks(ranks: List[float]) -> List[float]:
    if not ranks:
        return []
    if len(ranks) == 1:
        return [1.0]

    min_r = min(ranks)
    max_r = max(ranks)

    # BM25 ranks in SQLite FTS5 are negative (more negative = better)
    if abs(max_r - min_r) < 1e-9:
        return [1.0 for _ in ranks]

    normalized = []
    for r in ranks:
        norm = (max_r - r) / (max_r - min_r)
        normalized.append(norm)
    return normalized


def compute_filename_score(file_name: str, terms: List[str]) -> float:
    if not file_name or not terms:
        return 0.0
    fn_lower = file_name.lower()
    for term in terms:
        if term.lower() in fn_lower:
            return 1.0
    return 0.0


def compute_temporal_score(modified_at: str, time_range: Optional[TimeRange]) -> float:
    if not time_range or not time_range.resolved or not time_range.start_ts or not time_range.end_ts:
        return 0.0

    dt_mod = parse_iso_utc(modified_at)
    dt_start = parse_iso_utc(time_range.start_ts)
    dt_end = parse_iso_utc(time_range.end_ts)

    if not dt_mod or not dt_start or not dt_end:
        return 0.0

    if dt_start <= dt_mod <= dt_end:
        return 1.0

    # Near boundary (+/- 7 days)
    margin = 7 * 86400
    if (dt_start.timestamp() - margin) <= dt_mod.timestamp() <= (dt_end.timestamp() + margin):
        return 0.5

    return 0.0


def rank_candidates(
    candidates: List[Dict[str, Any]],
    parsed_query: ParsedQuery,
    time_range: Optional[TimeRange],
) -> List[Tuple[Dict[str, Any], float, str]]:
    """
    Ranks candidates and returns a list of tuples: (candidate, score, why_string).
    """
    if not candidates:
        return []

    # Collect raw ranks ONLY from candidates that actually matched FTS
    fts_matched_candidates = [c for c in candidates if c.get("fts_matched")]
    fts_raw_ranks = [float(c.get("rank") or 0.0) for c in fts_matched_candidates]
    fts_norm_list = normalize_fts_ranks(fts_raw_ranks)

    fts_norm_map: Dict[int, float] = {}
    for idx, c in enumerate(fts_matched_candidates):
        art_id = int(c["id"])
        fts_norm_map[art_id] = fts_norm_list[idx]

    scored: List[Tuple[Dict[str, Any], float, str]] = []

    for c in candidates:
        art_id = int(c["id"])
        fts_s = fts_norm_map.get(art_id, 0.0)
        sem_s = max(0.0, float(c.get("semantic_score") or 0.0))
        fn_s = compute_filename_score(str(c.get("file_name") or ""), parsed_query.terms)
        temp_s = compute_temporal_score(str(c.get("modified_at") or ""), time_range)
        episodes = c.get("episodes") or []
        memories = c.get("memories") or []
        ep_s = 1.0 if episodes else 0.0
        mem_s = 1.0 if memories else 0.0

        total_score = (
            WEIGHT_FTS * fts_s
            + WEIGHT_SEMANTIC * sem_s
            + WEIGHT_FILENAME * fn_s
            + WEIGHT_TEMPORAL * temp_s
            + WEIGHT_EPISODE * ep_s
            + WEIGHT_MEMORY * mem_s
        )

        why_parts = []
        # Check filename match
        for term in parsed_query.terms:
            if term.lower() in str(c.get("file_name") or "").lower():
                why_parts.append(f"Matched '{term}' in filename.")
                break

        # Check content match (FTS or Semantic)
        snippet = str(c.get("snippet") or "")
        mime = str(c.get("mime_type") or "").lower()
        content_matched = False
        if snippet and not any(term.lower() in str(c.get("file_name") or "").lower() for term in parsed_query.terms):
            for term in parsed_query.terms:
                if term.lower() in snippet.lower():
                    if mime.startswith("image/"):
                        why_parts.append(f"Matched '{term}' in OCR text from screenshot.")
                    else:
                        why_parts.append(f"Matched '{term}' in document text.")
                    content_matched = True
                    break

        if sem_s > 0.0 and not content_matched and not any(term.lower() in str(c.get("file_name") or "").lower() for term in parsed_query.terms):
            if mime.startswith("image/"):
                why_parts.append(f"Semantic match to concepts in OCR text from screenshot (similarity: {sem_s:.2f}).")
            else:
                why_parts.append(f"Semantic match to concepts in document text (similarity: {sem_s:.2f}).")

        # File type match
        if parsed_query.file_type:
            if parsed_query.file_type in mime or parsed_query.file_type in str(c.get("file_name") or "").lower():
                why_parts.append(f"File type matches '{parsed_query.file_type}'.")

        # Temporal match
        if temp_s == 1.0:
            if time_range and time_range.precision == "approximate":
                why_parts.append(f"Modified within approximate range of {time_range.original_expression}.")
            elif time_range:
                why_parts.append(f"Modified within requested range ({time_range.original_expression}).")
        elif temp_s == 0.5 and time_range:
            why_parts.append(f"Modified near requested range ({time_range.original_expression}).")

        # Episode / Activity match
        if parsed_query.intent == "activity" and episodes:
            why_parts.append(f"Artifact was active during episode on {str(c.get('modified_at') or '')[:10]}.")
        elif episodes:
            first_ep = episodes[0]
            why_parts.append(f"Part of episode '{first_ep.get('title')}' ({first_ep.get('start_ts', '')[:16]}–{first_ep.get('end_ts', '')[:16]}).")

        # Memory match
        if memories:
            first_mem = memories[0]
            why_parts.append(f"Topic matches associated memory '{first_mem.get('title')}'.")

        why_str = " ".join(why_parts) if why_parts else "Artifact matched search criteria."
        scored.append((c, total_score, why_str))

    # Sort descending by score
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
