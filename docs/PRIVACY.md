Privacy & Local-First Architecture

Principles
- Default local-only: nothing leaves the device unless the user explicitly opts in.
- Granular consent: the user picks which folders/sources to index.
- Transparent data model: the UI must display what is indexed and why.
- Easy data control: the user can delete events, episodes, and memories and wipe the index.

What the system collects (by default)
- Event metadata: timestamps, event type, canonicalized paths, app names.
- Artifact metadata & cached extracts: text extracts, OCR results, thumbnails (cached locally).
- Aggregates (episodes and memories) derived from events.
- Interaction logs (clicks/pins) stored locally to improve ranking.

Where data is stored
- Local app data directory (platform-specific): indexed metadata (SQLite), cached extracts (app cache), vector index files (FAISS/LMDB).
- Optional: user-enabled encryption of cache using OS keystore or a passphrase.

No cloud default
- Core product functions with zero network calls and no cloud account. Optional cloud features must be explicit, opt-in, zero-knowledge when possible.

Privacy features
- Selective indexing: user chooses folders and connectors.
- Preview & confirm: before initial index, show sample files that will be processed.
- Audit & provenance UI: shows which events produced a memory and what artifacts were used.
- Secure deletion: permanently remove events/episodes/memories and optionally shred cached extracts.
- Encrypted cache: optional AES-256 encryption of cached extracts and vectors; keys stored in OS keystore (Windows Credential Manager / macOS Keychain).
- Telemetry: minimal, opt-in, and aggregated with differential privacy if enabled.

Data export & portability
- Allow export of Memories/Episodes/Events as JSON bundles to allow backups and migration.

Data minimization
- Keep raw payloads (e.g., full HTTP headers or attachments) only as long as needed; default retention can be short for raw payloads and long for enriched metadata.

Auditability & user control
- Provide UI for searching and listing all events and the ability to delete single events or bulk by time range.
- Provide a "what did I share" audit page if cloud sync is enabled.

End of PRIVACY.md