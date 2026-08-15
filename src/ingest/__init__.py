"""Ingestion orchestration layer for Life Search.

Wires the existing Events -> Episodes -> Memories pipeline into the
real indexing path so search can actually enrich results.

This module does NOT reimplement any of the underlying components; it
only composes their existing, tested APIs (FilesystemEventSource,
EventStore, EpisodeEngine, MemoryBuilder, EpisodeStore, MemoryStore).
"""
