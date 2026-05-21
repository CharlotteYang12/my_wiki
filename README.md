# LLM Wiki

This folder is a starter implementation of Karpathy's LLM Wiki pattern.

## How To Use

1. Put source documents into `raw/`.
2. Ask the agent to ingest one source, for example:

   ```text
   Ingest raw/my-article.md into the wiki.
   ```

3. Read the generated pages under `wiki/`.
4. Ask questions against the wiki as it grows.

## Key Files

- `llm-wiki.md`: original idea document.
- `AGENTS.md`: rules the agent should follow when maintaining this wiki.
- `wiki/index.md`: catalog of generated wiki pages.
- `wiki/log.md`: chronological record of work done.
