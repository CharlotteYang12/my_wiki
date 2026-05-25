# Wiki Log

Append-only activity log for ingests, saved queries, and maintenance passes.

## [2026-05-14] setup | Initialize LLM Wiki

- Created the initial wiki directory structure.
- Added `AGENTS.md` to define maintenance rules.
- Added `wiki/index.md` and `wiki/log.md`.

## [2026-05-16] maintenance | Rewrite giant company financing page

- Converted `wiki/巨头公司融资.md` from comma-separated notes into a concise Markdown table.
- Updated `wiki/index.md` with the maintained page entry.

## [2026-05-16] maintenance | Verify commercial data

- Checked public sources for valuation, financing, revenue, order, production, and IPO claims in `wiki/巨头公司融资.md`.
- Rewrote the page as a verification table with source links and confidence notes.

## [2026-05-16] maintenance | Restructure financing table

- Removed verification and correction columns from `wiki/巨头公司融资.md`.
- Split technical route and product information into standalone columns.

## [2026-05-16] maintenance | Normalize valuation currency

- Bolded company valuation figures in `wiki/巨头公司融资.md`.
- Converted RMB valuation references to approximate USD values.

## [2026-05-16] maintenance | Add financing stages

- Added a financing stage column to `wiki/巨头公司融资.md`.

## [2026-05-25] ingest | DreamerV4 notes

- Ingested `raw/assets/dreamerv4.md` into `wiki/sources/dreamerv4.md`.
- Added concept pages: `wiki/concepts/World Model Imagination Training.md`, `wiki/concepts/PMPO.md`, and `wiki/concepts/Shortcut Models.md`.
- Added entity page: `wiki/entities/DreamerV4.md`.
- Updated `wiki/index.md` to include the new source, concepts, and entity links.

## [2026-05-25] maintenance | Consolidate wiki images

- Moved all wiki image files into `wiki/assets/`.
- Updated image references in `wiki/bfm-zero.md`, `wiki/agent-react.md`, and `wiki/dreamerv4.md` to `assets/...` paths.
