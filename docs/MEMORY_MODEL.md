Memory Model — Events, Episodes, Memories

Overview

This document defines the core domain model: Event, Episode, Memory, and related concepts. These are the canonical abstractions for V2 and must be treated as first-class entities in storage, retrieval, and UI.

Definitions

Artifact
- Any persisted object on disk or external to the event stream that can serve as evidence: files, screenshots, browser pages, audio, code, emails.
- Contains metadata: path, mime_type, size, content_hash, created_at, modified_at, source_app.

File
- A subtype of Artifact stored on disk (PDF, DOCX, TXT, code file).

Screenshot
- An image artifact captured by the OS or user. Usually stored in a screenshots folder; evidence often requires OCR.

Event
- A normalized observation about something that happened on the user's computer at a point in time.
- Example types: FileCreated, FileModified, FileOpened, FileDeleted, ScreenshotCaptured, AppOpened, BrowserVisited, DownloadCompleted, GitCommit, TerminalCommandExecuted, EditorSave, SearchPerformed.
- Core fields: id, type, timestamp (UTC ms), source (capture subsystem), artifact_id (nullable), app (optional), raw_payload (JSON), confidence.
- Events are immutable once recorded; updates create new events referencing the earlier event if needed.

Episode
- A bounded set of Events that together represent one continuous user activity or task.
- Example: a 90-minute coding session that included browsing a research article, downloading a PDF, editing code, and creating a screenshot.
- Episode fields: id, start_ts, end_ts, event_ids (ordered), dominant_apps, dominant_entities, inferred_title, confidence, project_hint.
- Episodes can be overlapping (user switches tasks) or hierarchical (a larger Episode containing smaller sub-Episodes).

Memory
- A higher-level, human-focused abstraction derived from one or more Episodes (or parts of episodes) intended for later recall.
- Example: "You researched Qdrant and updated retrieval.py on 2026-08-05 between 20:00 and 22:00." This Memory links to the Episode, supporting Events, and artifacts.
- Memory fields: id, summary_text, start_ts, end_ts, linked_episode_ids, linked_event_ids, evidence_references (artifact ids), entities, people, confidence_score, provenance.

Entity
- Named concepts extracted from content and events: people, organizations, technologies, project names, error codes, function names.
- Entities are used for clustering, queries, and project detection.

Project
- A user-visible grouping that persists across time: either user-defined (explicit project) or auto-inferred from event co-occurrence (e.g., files under ~/projects/jyotishai and editor sessions that touch those files).

Activity
- Short, transitory user interactions like opening an app or visiting a page; often represented as a sequence of Events and used for Episode detection.

Relationship
- Typed edge representing a link between domain objects (Event→Artifact, Episode→Event, Memory→Episode, Entity→Document).
- Relationships record: from_id, to_id, type, evidence (list of event ids), confidence.

Source
- The origin of an event or artifact: OS watcher, browser connector, git hook, manual import. Source records connection details and consent.

Evidence
- A minimal, verifiable pointer used to justify an inference. Could be a particular Event (download event), artifact (file path + bytes hash), or textual snippet (OCR text containing the token that matched a query).
- Evidence items include a confidence score and optional human-readable explanation.

Relationships and provenance
- Every inferred relationship must carry evidence and a confidence score.
- Provenance must allow reconstructing the chain: Memory → Episode → Events → Artifact(s) with timestamps and links to raw payloads.

Data model notes (conceptual)
- Store Events append-only for auditability.
- Store Episodes as first-class objects that reference Events.
- Memories are derived stores used for UI and recall; they can be materialized lazily.
- Use explicit relationship tables to avoid ad-hoc references embedded in text fields.

Examples
- Event sequence: BrowserVisited(page: Qdrant article) -> DownloadCompleted(file: qdrant.pdf) -> AppOpened(VSCode) -> FileModified(retrieval.py) -> ScreenshotCaptured(error-mongo)
- Episode: group those events into "Qdrant research and retrieval implementation" with start_ts of the BrowserVisited and end_ts of the ScreenshotCaptured.
- Memory: "Researched Qdrant and updated retrieval.py; saved related PDF and took MongoDB error screenshot." Links to episode id, download event id, file id, screenshot id.

End of MEMORY_MODEL.md
