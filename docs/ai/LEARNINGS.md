# LEARNINGS

Team-shared lessons, maintained by /uexel:learn via uexel-scribe. Dated one-liners with
the why. Only lessons a teammate benefits from — personal notes belong to auto-memory.

- 2026-08-02: `ON CONFLICT ON CONSTRAINT <name>` fails against a TypeORM
  `@Index(..., { unique: true })` — that decorator creates a plain unique index, not a
  table constraint, and Postgres only resolves `ON CONFLICT ON CONSTRAINT` against an
  actual constraint catalog entry (confirmed live against idx_observation_dedupe:
  "constraint ... does not exist"). Use `ON CONFLICT (col1, col2, ...) DO NOTHING`
  instead — the column-list form works against any unique index.
