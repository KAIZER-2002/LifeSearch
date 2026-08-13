Event Model — Normalized Event Schema and Capture

Purpose

A clear, extensible, and concise Event model is the foundation of the memory architecture. Events are immutable records of activity; they enable episode detection, provenance, and evidence.

Core event schema (conceptual)
- id: string (UUID)
- type: string (enum) — standardized list of event types (see below)
- timestamp: integer (UTC ms)
- source: string — capture subsystem id (filesystem-watcher, screenshot-watcher, browser-history, git-hook)
- artifact_id: string|null — pointer to an Artifact when applicable
- app: string|null — application that generated or was foreground at the time (e.g., Chrome, VSCode, Terminal)
- raw_payload: JSON — raw data captured from OS or connector (kept for audit and re-enrichment)
- metadata: JSON — normalized fields for quick filtering (path, mime_type, url, command, window_title)
- confidence: float [0.0..1.0] — how confident the capture/enrichment is (OCR/NER may lower confidence)
- tags: [string] — fast labels (e.g., error, download, screenshot, commit)
- linked_entities: [entity_id] — optional links to extracted entities

Event types (initial list)
- FileCreated
- FileModified
- FileOpened
- FileDeleted
- FileRenamed
- FileDownloaded
- ScreenshotCaptured
- AppOpened
- AppClosed
- WindowFocused
- BrowserVisited (url, title)
- ClipboardPasted
- TerminalCommandExecuted
- GitCommit
- EditorSave
- SearchPerformed
- UserAnnotated (manual note)

Source and capture details
- Each source connector must declare its identity and permissions required.
- Sources are responsible for creating the Event.raw_payload and initial metadata.
- The capture layer should not perform heavy enrichment synchronously; emit the raw Event and schedule enrichment in the background.

Metadata normalization
- Normalize timestamps to UTC ms.
- Normalize paths to absolute canonical paths and compute content_hash where applicable.
- Map common app names to canonical forms ("Google Chrome" -> "Chrome").

Confidence model
- Raw events from OS watchers are high confidence for timing and file path, lower for content (until extraction runs).
- OCRed text gets a per-event OCR confidence score that affects evidence weight.

Privacy boundary
- Events contain potentially sensitive info. Default policy: store only what is necessary for indexing and provenance. Keep raw_payload encrypted if the user opts in. Do not transmit Events.

Event immutability and updates
- Events are append-only. If new information is discovered (e.g., better OCR result), a new Event or a linked enrichment record should be written rather than mutating the original Event. This preserves provenance.

Event retention and pruning
- Provide user controls for retention windows for event-level details (e.g., keep raw payload for 90 days, keep metadata but drop raw payload earlier).

Example Event JSON (download)
{
  "id": "evt-uuid-1",
  "type": "FileDownloaded",
  "timestamp": 1712750400000,
  "source": "browser-chrome",
  "artifact_id": "art-uuid-1",
  "app": "Chrome",
  "raw_payload": { "url": "https://example.com/qdrant.pdf", "referer": "https://blog.example/" },
  "metadata": { "path": "C:\\Users\\You\\Downloads\\qdrant.pdf", "mime_type": "application/pdf" },
  "confidence": 0.99,
  "tags": ["download","pdf"],
  "linked_entities": ["ent-qdrant"]
}

End of EVENT_MODEL.md
