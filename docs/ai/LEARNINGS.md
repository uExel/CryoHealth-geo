# LEARNINGS

Team-shared lessons, maintained by /uexel:learn via uexel-scribe. Dated one-liners with
the why. Only lessons a teammate benefits from — personal notes belong to auto-memory.

- 2026-08-02: `ON CONFLICT ON CONSTRAINT <name>` fails against a TypeORM
  `@Index(..., { unique: true })` — that decorator creates a plain unique index, not a
  table constraint, and Postgres only resolves `ON CONFLICT ON CONSTRAINT` against an
  actual constraint catalog entry (confirmed live against idx_observation_dedupe:
  "constraint ... does not exist"). Use `ON CONFLICT (col1, col2, ...) DO NOTHING`
  instead — the column-list form works against any unique index.
- 2026-08-03: psycopg3 returns Postgres `uuid` columns as Python `uuid.UUID` objects,
  not strings. That's fine as a SQL bind parameter (psycopg adapts it), but it breaks
  silently downstream if the same value later needs to go into a JSON payload —
  `json.dumps`/httpx has no default encoder for `UUID` and raises `TypeError`. Cast to
  `str()` at the boundary where a DB-sourced id crosses into an HTTP request body.
- 2026-08-03: an unordered SQL query feeding a "closest observation to date X" lookup
  (via Python `min()`/`max()` over rows) is non-deterministic when two rows tie for
  closest — which row wins depends on whatever order Postgres happens to return them
  in, not on anything meaningful. Confirmed live: this picked a genuinely noisy outlier
  reading over its more representative same-distance neighbor, producing a spurious 19x
  "growth" figure. Always add `ORDER BY` to a query whose result order feeds tie-breaking
  logic, even when the query "shouldn't" care about order.
