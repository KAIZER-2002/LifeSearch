# AI Agent Guide for Automated Code Assistance

Purpose
- Supply a strict, prescriptive playbook for how an AI coding agent ("agent") must behave when making changes to this repository.
- Ensure safety, privacy, reproducibility, and auditability of all automated changes.

Scope and audience
- This document is for automated agents that will open branches/PRs, modify code, run tests, and update docs.
- Human maintainers can use it to verify agent behavior and to audit decisions.

Agent identity and required metadata
- Every commit or PR created by an agent MUST include an identity block in the PR body and an abbreviated identity in the commit message. The block MUST contain:
  - agent_name: e.g., `ai-code-agent/v1`
  - runtime: e.g., `Copilot CLI runtime in VS Code`
  - model: pinned model identifier and version used (e.g., `gpt-4o-code-2026-08`)
  - timestamp: ISO 8601 UTC
  - decision_summary: one-paragraph rationale describing what changed and why
- Example for PR body and/or commit message:
  ```
  Agent: ai-code-agent/v1
  Runtime: Copilot CLI runtime in VS Code
  Model: gpt-4o-code-2026-08 (pinned)
  Time: 2026-08-11T23:21:10Z
  Decision summary: Fixed NPE in parser by normalizing input; added unit tests for edge cases.
  ```

High-level architecture principles
- Make the smallest effective change: prefer small, isolated PRs.
- Preserve public APIs and backward compatibility by default.
- Keep module boundaries clear; do not move code between modules without an ADR (Architecture Decision Record).
- Document any non-trivial design decision in the PR and, when needed, create or update an ADR.

Core domain concepts
- Event: an immutable, timestamped record of a single observation or action. JSON-serializable with fields: id, type, timestamp, source, payload.
- Episode: a bounded sequence of events that represent one session or interaction. Episodes have metadata: id, start, end, participants.
- Memory: curated, durable facts derived from episodes. Memories store provenance: source_event_ids, confidence, created_by, timestamp.

Storage and format rules
- Use structured JSON for events/episodes/memories and include explicit schema versioning (e.g., schema_version: "v1").
- When modifying event schemas: add the new version, write a migration path, and preserve previous versions unless explicit deprecation is approved by maintainers.

Coding rules — what agents may change automatically
- Allowed automatic edits (agent may commit directly):
  - Fixing small bugs accompanied by tests demonstrating the fix.
  - Refactoring internal functions with no public API change, limited to a few files.
  - Adding unit/integration tests and test helpers.
  - Small, well-scoped performance improvements with benchmark evidence.
  - Adding or updating docs that are directly related to the change.
- Prohibited automatic edits (require human approval):
  - Changes to public/external APIs, protocols, or data contracts.
  - Security-sensitive code: authentication, authorization, encryption, key management.
  - Any change that enables new remote data exfiltration, telemetry, or third-party network calls.
  - Large-scale refactors (more than N files or touching core modules — treat N conservatively, e.g., 5-7 files).
  - Adding infrastructure or deployment config changes (CI, Docker, Kubernetes) without maintainer approval.

Coding hygiene
- Run the project's linter(s) and formatting tools; PR must be lint-clean.
- Do not introduce new warnings, and do not change linting rules in the same PR as functional changes.
- Keep commits small and focused; prefer multiple focused commits over a single large one. Squash commits before merge only if history readability is preserved.

Testing rules
- Tests must be deterministic. Use mocks for external services.
- For functional changes: include unit tests that cover edge cases and at least one integration test where appropriate.
- Run the full test suite (or the CI-targeted test subset) locally before creating a PR.
- Add test instructions to the PR body explaining how to run the tests and expected outputs.
- Do not disable, mark flaky, or skip existing tests in an automated change without opening a separate issue and getting human approval.

Documentation rules
- For any user-visible or API change, update README, docs/, and inline docstrings.
- Add a concise CHANGELOG entry for behavior changes, in the repository's canonical changelog file.
- Include an "Agent rationale" section in the PR body that explains the why, risks, and validation steps.

Dependency rules
- Prefer minimal, well-maintained, permissively-licensed dependencies.
- Automated dependency bumps are allowed only for patch releases and must include: test results, security notes, and a short risk analysis.
- Minor or major upgrades must be separate PRs, include a compatibility plan and a test matrix.
- Do not add network-only SDKs or telemetry libraries without maintainer approval.

Privacy and security rules
- Never commit secrets, test tokens, or PII into the repo. If an accidental secret commit happens, open an incident and follow the repo incident response instructions.
- Do not add remote logging or telemetry that transmits user content without explicit human-approved UX and documentation.
- For connectors that access external accounts, require explicit opt-in UI and a consent flow; document retention, scope, and revocation.

Model and reproducibility rules
- Pin the model and model version used for any decision-making.
- Save and include (in PR body) the exact prompt template and hyperparameters used if they materially influence outputs.
- Where randomness can affect results (tests, generation), include deterministic seeds and make them configurable for CI reproducibility.

Adding new extraction connectors — exact process
1. Create directory: src/connectors/<connector-name>/
   - Add README.md with purpose, required permissions, and privacy considerations.
   - Add src/<connector>.{js,py,...} implementing the connector interface.
   - Export a factory or registration function compatible with the project's connector registry.
2. Implement integration boundary: connector emits raw Events adhering to the Event schema version and does not perform remote writes by default.
3. Tests:
   - Unit tests that mock external APIs and assert Event JSON correctness.
   - Integration tests that validate ingestion into local pipelines using sample (sanitized) fixtures.
4. Configuration and consent:
   - Add configuration option and onboarding consent UI (or a documented CLI/README step) describing what data is accessed and retention policy.
5. Docs:
   - Add docs/connectors.md entry and update docs/EVENT_MODEL.md.
6. Security/privacy: Add a privacy checklist item to the PR showing credentials handling and that no secrets are stored in repo.
7. CI: Ensure connector tests run in CI with mocks; do not commit real tokens to CI.

Definition of Done (strict checklist for PRs/changes)
- [ ] Agent identity block present in commit message and PR body (model, runtime, timestamp, decision_summary).
- [ ] Changes are minimal and scoped to the stated goal; PR modifies only required files.
- [ ] Linter and formatting checks pass locally and in CI.
- [ ] All new and affected existing tests pass locally and in CI.
- [ ] PR body contains a reproduction/validation section with exact commands and expected results.
- [ ] Documentation updated for user-visible changes, including changelog entry.
- [ ] Privacy checklist completed (no secrets/PII committed, consent flows added if needed).
- [ ] Dependency changes are explained and limited to patch-level unless an upgrade PR.
- [ ] Human approval: at least one human reviewer has approved the PR. If the change affects security/infra, the appropriate security reviewer must approve.
- [ ] Rollback plan included for runtime-impacting changes.
- [ ] Audit entry: an audit artifact (log file or database entry) records agent_id, files_changed, tests_run, and CI status — attached to the PR or referenced in the body.

Merging and post-merge responsibilities
- Merge after CI green and required approvals.
- After merging, monitor the next deploy and runtime errors for 24-72 hours depending on impact. If issues arise, revert or apply the rollback plan.
- Update release notes and tag releases per repo convention.

Failure modes and escalation
- If tests fail or results are non-deterministic: do not merge; open an issue and mark the PR `draft/needs-human`.
- If the change touches security-sensitive code or unintentionally transmits data externally: immediately notify maintainers, create an incident issue, and pause further agent edits.

Auditability and retention
- Keep an auditable trace of agent decisions. Save prompts, model versions, and test logs in the PR or a linked artifact store.
- Keep memory and event schema migrations reversible and documented.

When to stop and ask humans
- Any change that affects API contracts, security, infra, or privacy beyond a small, well-understood fix.
- Any non-trivial migration that requires data transformation.
- Any ambiguous behavior where the agent cannot write a deterministic test to validate the change.

Quick reference (Do / Don't)
- DO: Make small, tested, documented changes; include agent metadata; run CI locally; add migration scripts for schema changes.
- DO: Mock external calls in tests and keep connectors opt-in.
- DON'T: Commit secrets or PII; add telemetry or external syncing without consent; change security-sensitive code without human sign-off.

This guide is authoritative. Automated agents must follow these rules exactly unless an explicit, documented, and human-approved exception is granted. Keep this document up to date. When you update it, add a short agent-facing changelog entry at the top with timestamp, author (agent or human), and reason.
