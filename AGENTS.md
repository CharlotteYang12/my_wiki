# LLM Wiki Agent Guide

This workspace is a persistent markdown wiki maintained by an LLM agent.

## Directory Layout

- `raw/`: immutable source material supplied by the user.
- `raw/assets/`: images and attachments referenced by raw sources.
- `wiki/`: LLM-maintained knowledge base.
- `wiki/sources/`: one summary page per ingested source.
- `wiki/concepts/`: topic and concept pages.
- `wiki/entities/`: people, organizations, projects, places, and named entities.
- `wiki/questions/`: synthesized answers worth preserving as wiki pages.
- `wiki/index.md`: content-oriented catalog of wiki pages.
- `wiki/log.md`: append-only chronological activity log.

## Operating Principles

1. Treat `raw/` as read-only source of truth.
2. Write and update pages only under `wiki/`, unless the user asks otherwise.
3. Prefer concise markdown with Obsidian-style internal links like `[[Page Name]]`.
4. When adding or changing wiki pages, update `wiki/index.md`.
5. Append a dated entry to `wiki/log.md` for every ingest, query saved to the wiki, or maintenance pass.
6. Preserve uncertainty. Mark unclear claims as open questions rather than inventing missing facts.
7. Note contradictions explicitly and link to the source pages that disagree.

## Ingest Workflow

When the user asks to ingest a source:

1. Read the source from `raw/`.
2. Create or update a summary page in `wiki/sources/`.
3. Extract important concepts and entities into `wiki/concepts/` and `wiki/entities/`.
4. Add cross-links between related pages.
5. Update `wiki/index.md`.
6. Append an entry to `wiki/log.md`.

## Query Workflow

When answering questions about the wiki:

1. Read `wiki/index.md` first.
2. Read the relevant wiki pages.
3. Answer with citations to file paths when useful.
4. If the answer is reusable, ask whether to save it under `wiki/questions/`.

## Maintenance Workflow

Periodically check for:

- orphan pages
- missing index entries
- stale claims
- contradictions
- important concepts without their own page
- source summaries that lack links to concept or entity pages
