"""Feedback capture for Life Search (Phase C4).

Isolated, local-first storage of user ranking signals (click / ignore / pin)
so a later milestone can consume them for personalization. Does not modify
the core retrieval architecture.
"""

from .store import FeedbackStore, VALID_ACTIONS

__all__ = ["FeedbackStore", "VALID_ACTIONS"]
