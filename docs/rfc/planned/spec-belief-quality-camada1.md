# SPEC: Belief quality — Camada 1 (higiene de belief: eliminar degenerados)

**Status:** planned · **Created:** 2026-10-15 · **Fork-only (maturing) · learning-loop Camada 1 (higiene)**
**Author:** Claudio + Zero (G0: seven) · **ARC:** f0e2c306 · **HEAD:** 83be6d25 (main)
**Depends on:** R1 ontology intake (`MCP_BELIEF_ONTOLOGY_FILTERING`, belief_service.py:566-620) — DONE, do not touch.

## Problem

Beliefs are created by copying a group representative **verbatim** — `derive_beliefs`
inserts `group_content` with no synthesis (belief_service.py:413, create path at
belief_service.py:466-474; `group_content` is the raw cluster representative built in
`_group_observations`, belief_service.py:755-826). The intake noise filter `_is_noise`
(belief_service.py:50) rejects by: length `<15` chars or `<3` words, prefix list,
session/UI regex patterns, length `>500`, bullet-ratio `>0.6`, and (separately, in
`derive_beliefs`) entropy `<=0.25` via `_content_entropy` (belief_service.py:83).

**What that filter does NOT catch: truncated/incomplete sentences and orphan markdown
headings.** A fragment of ≥15 chars and ≥3 words that ends in a dangling conjunction,
preposition or article — or is only a bold heading with no body — passes every current
check and becomes a belief verbatim, polluting the injection top-k.

### Baseline (measured on the live corpus, `~/local-data/mcp/sqlite_vec.db`, 15/out)

Beliefs by status: **active=360**, candidate=1997, superseded=886, inactive=6.

- **active (360):** 26 (**7,2%**) are already filterable by the *current* rule but
  present as historical debt (created before the filter existed); **+6 (1,7%)**
  degenerates that **pass** the current filter.
- **candidate (1997):** 52 filterable + **39 (2,0%)** degenerates that **pass**.

Real passing-degenerate examples (must be caught by the new rule):

- Truncated sentence (ends in preposition/conjunction): `Conflitos recorrentes são` ·
  `Se o fato mudou, o` · `Ao avaliar regress`
- Orphan markdown heading (bold title, no body sentence after): `**Causa Raiz Identificada:**` ·
  `**Proxy Conecta.gov (` · `**Ativação OSS do Greptile:**` · `**QueryFlow Audit Design**` ·
  `**Memory Service (F022)**`

(Markers like `TYPE:` / `convention` / `learning` are already caught by the `<15` chars /
`<3` words rule — not in scope to re-handle.)

## Objective

Give the belief pipeline a **degenerate-fragment reject rule** that catches truncated
sentences and orphan headings, applied in **two places**:

- **1a — intake:** harden `_is_noise` so a new degenerate never becomes a belief.
- **1b — backlog:** a one-shot, opt-in, capped, idempotent pass that marks
  `status='superseded'` the beliefs already in the table that the new rule rejects —
  preserving provenance (no delete).

No synthesis is introduced (that is Camada 2). No new DDL.

## Requirements (EARS)

### 1a — Intake reject rule (harden `_is_noise`, belief_service.py:50)

- **R1a-1 (dangling stopword):** WHEN the candidate content's last alphabetic token
  (ignoring trailing whitespace and a single trailing `.`/`…` is absent) is a cut-stopword
  from the pt+en conjunction/preposition/article list (see RÉGUA), THE SYSTEM SHALL
  classify the content as noise and reject it.
- **R1a-2 (dangling punctuation):** WHEN the candidate content, after right-stripping
  whitespace, ends in `:`, `,`, `;`, `-`, `–`, `—`, `/`, `(`, `[`, or `{`, THE SYSTEM
  SHALL classify the content as noise and reject it.
- **R1a-3 (unbalanced opener):** WHEN the candidate content contains more opening than
  closing brackets of any kind (`(`>`)`, `[`>`]`, `{`>`}`), THE SYSTEM SHALL classify the
  content as noise and reject it.
- **R1a-4 (orphan markdown heading):** WHEN the candidate content, after stripping leading
  `#`/`*`/whitespace, matches the orphan-heading regex (see RÉGUA) — i.e. it is a single
  bold/heading line (optionally ending in `:`) with no non-empty body line after it — THE
  SYSTEM SHALL classify the content as noise and reject it.
- **R1a-5 (invariant, legitimate short sentence):** WHEN the candidate content ends in a
  content word (noun/verb/adjective — i.e. NOT a cut-stopword) followed by terminal
  punctuation `.`/`!`/`?`, and has balanced brackets, THE SYSTEM SHALL NOT reject it on
  grounds of R1a-1..R1a-4 (short legitimate sentences survive).
- **R1a-6 (opt-in, reversible):** THE SYSTEM SHALL gate the new reject rule behind the
  flag `MCP_BELIEF_DEGENERATE_FILTER` (default OFF); WHEN disabled, `_is_noise` behaviour
  is byte-identical to today.
- **R1a-7 (compose, do not replace):** THE SYSTEM SHALL apply the new rule in addition to
  all existing `_is_noise` checks; it SHALL NOT remove or weaken any current check.

### 1b — Backlog cleanup (one-shot opt-in pass)

- **R1b-1 (opt-in):** THE SYSTEM SHALL expose a one-shot cleanup entry point that runs
  ONLY when flag `MCP_BELIEF_DEGENERATE_SWEEP` is set (default OFF); WHEN unset, it is a
  no-op returning zero changes.
- **R1b-2 (apply same rule):** WHEN the sweep runs, THE SYSTEM SHALL evaluate every belief
  with `status IN ('active','candidate')` against the exact same reject rule used at intake
  (R1a-1..R1a-4), guaranteeing intake and sweep cannot diverge.
- **R1b-3 (supersede, not delete):** WHEN a belief is rejected by the rule, THE SYSTEM
  SHALL set its `status='superseded'` and `updated_at=now` and SHALL NOT delete the row or
  alter `derived_from`/`contradicted_by` (provenance preserved).
- **R1b-4 (cap):** THE SYSTEM SHALL mark at most `MCP_BELIEF_DEGENERATE_SWEEP_CAP`
  (default 1000) beliefs per run, bounding blast radius.
- **R1b-5 (idempotent):** WHEN the sweep is run a second time with no intervening changes,
  THE SYSTEM SHALL mark zero additional beliefs (already-superseded rows are skipped;
  re-running is byte-safe).
- **R1b-6 (report):** THE SYSTEM SHALL return a stats dict `{"evaluated": N, "superseded":
  M, "capped": bool, "errors": [...]}` for observability.

## Invariant — legitimate short sentences MUST survive

The rule is conservative: a short, complete sentence ending in a content word + terminal
punctuation is **valid** and MUST NOT be rejected. Canary sentences (the filter must keep
all of these; see RÉGUA for why each passes):

1. `Nunca deixar commit sem push.` — ends in noun `push` + `.`
2. `Usar sempre LIMIT em queries PostgreSQL.` — ends in noun `PostgreSQL` + `.`
3. `O token em ~/.git-credentials expira.` — ends in verb `expira` + `.`
4. `Preferir ONNX ao invés de torch no CPU.` — ends in noun `CPU` + `.`

(Note: `O token ... expira.` deliberately contains the article `O` at the *start* and the
preposition `em`/`de` *mid-sentence* — the rule only inspects the **final** token, so these
do not trigger R1a-1.)

## Out of scope

- **Real synthesis** (generate a generalization instead of copying `group_content`) =
  **Camada 2**, `rfc-belief-synthesis.md` — different arc.
- **R1 ontology intake** (`MCP_BELIEF_ONTOLOGY_FILTERING`) = already done, untouched here.
- Recall@pool 0.30→0.8 = Camada 2 target, not this layer.
- Re-ranking / specificity scoring = learning-loop frente A.

## Acceptance criteria (measurable — test against a COPY of the live corpus, never live)

- [ ] **AC1 (degenerate drop, active):** GIVEN the live `active` corpus (360), WHEN the
  1b sweep runs with the flag on, THEN the proportion of degenerates among `active` beliefs
  decreases — the ~1,7% (6) that currently pass the filter PLUS the ~7,2% (26) historical
  debt are moved to `superseded` (combined ≈8,9% of active removed from the active set).
- [ ] **AC2 (degenerate drop, candidate):** GIVEN the live `candidate` corpus (1997), the
  sweep supersedes the 39 (2,0%) passing degenerates + the 52 historical-debt filterable.
- [ ] **AC3 (canaries survive):** GIVEN the 4 canary sentences above, WHEN `_is_noise` runs
  with the degenerate filter ON, THEN each returns `False` (not noise) — asserted individually.
- [ ] **AC4 (no legitimate belief reproved):** GIVEN every current `active` belief that is
  NOT in the measured degenerate set, WHEN the rule is applied in a copy, THEN zero of them
  are flagged (false-positive rate = 0 on the live legitimate corpus). Measured by diffing
  the flagged set against the hand-verified degenerate list.
- [ ] **AC5 (idempotent):** WHEN the sweep runs twice on the copy, THEN the second run
  reports `superseded=0`.
- [ ] **AC6 (flags off = no-op):** WITH both flags off, `_is_noise` and the pipeline are
  byte-identical to HEAD 83be6d25 (regression test).
- [ ] **AC7 (no DDL):** No schema migration added; sweep uses existing `status` column.

## Notes / evidence (G0, seven, 15/out)

- Intervention point intake: `_is_noise` (belief_service.py:50) — add the rule as an extra
  guarded branch; `_content_entropy` (belief_service.py:83) stays as-is.
- `derive_beliefs` (belief_service.py:413) already calls `_is_noise` on each observation
  pre-grouping (belief_service.py:446-449) and entropy-filters at belief_service.py:453-458,
  so hardening `_is_noise` automatically protects intake with no call-site change.
- `create_singular_belief` (belief_service.py:~150) also calls `_is_noise` — the new rule
  covers the singular path for free.
- Supersede path for 1b: reuse `_set_belief_status(belief_hash, "superseded", now)`
  (belief_service.py:888) — exact same mechanism the floor-supersession already uses.
- The sweep is a NEW method (reg to implement) reading `SELECT belief_hash, content, status
  FROM beliefs WHERE status IN ('active','candidate')` and applying the rule; it adds no DDL.

---

## RÉGUA DE DEGENERADO (definição precisa e testável — fonte única para o rok)

A function `_is_degenerate_fragment(content: str) -> bool` SHALL return `True` iff ANY of
D1..D4 holds, applied to `text = content.strip()`. The existing `_is_noise` short-circuits
(`<15` chars, `<3` words, etc.) run FIRST; this rule only decides the remaining cases.

### Preprocessing
- `text = content.strip()`
- `last_line = text.splitlines()[-1].strip()` (for heading check, the whole block matters — see D4)
- Tokenization for D1: `tokens = re.findall(r"[A-Za-zÀ-ÿ'`-]+", text)`; `last_tok = tokens[-1].lower()` if any.
- "Ends in terminal punctuation" means `text` (right-stripped) ends in one of `.`, `!`, `?`, `…`.

### D1 — Dangling cut-stopword (truncated sentence)
`True` WHEN `last_tok` ∈ `CUT_STOPWORDS` **and** `text` does NOT end in a closing quote/bracket
that would make it a legitimate parenthetical. (A trailing `.` is irrelevant here: a sentence
that ends `...são.` is still degenerate if `são` is the last word — but in practice the real
degenerates end with NO terminal punctuation, so D1 fires on `last_tok` regardless of a trailing period.)

`CUT_STOPWORDS` (exact, lowercase):

**Portuguese — prepositions:** `de`, `da`, `do`, `das`, `dos`, `em`, `na`, `no`, `nas`,
`nos`, `a`, `à`, `ao`, `aos`, `às`, `para`, `pra`, `por`, `pelo`, `pela`, `pelos`, `pelas`,
`com`, `sem`, `sob`, `sobre`, `entre`, `até`, `desde`, `perante`, `ante`, `após`, `contra`,
`trás`.
**Portuguese — conjunctions:** `e`, `ou`, `mas`, `porém`, `contudo`, `todavia`, `que`,
`se`, `como`, `quando`, `porque`, `pois`, `logo`, `portanto`, `então`, `nem`, `enquanto`,
`embora`, `conforme`, `caso`.
**Portuguese — articles/determiners:** `o`, `a`, `os`, `as`, `um`, `uma`, `uns`, `umas`,
`este`, `esta`, `esse`, `essa`, `aquele`, `aquela`, `seu`, `sua`, `meu`, `minha`.

**English — prepositions:** `of`, `in`, `on`, `at`, `to`, `for`, `with`, `without`, `by`,
`from`, `into`, `onto`, `upon`, `about`, `over`, `under`, `between`, `among`, `through`,
`during`, `before`, `after`, `against`, `per`, `via`.
**English — conjunctions:** `and`, `or`, `but`, `nor`, `yet`, `so`, `because`, `that`,
`if`, `when`, `while`, `as`, `than`, `whether`, `although`, `though`, `since`, `unless`.
**English — articles/determiners:** `the`, `a`, `an`, `this`, `that`, `these`, `those`,
`its`, `their`, `his`, `her`, `my`, `our`, `your`.

> Note: `a`/`à`/`o`/`as`/`os`/`that`/`as` are intentionally in the list for the **final**
> token only. The invariant (R1a-5) is preserved because real sentences end in a content
> word, not one of these.

### D2 — Dangling punctuation
`True` WHEN `text.rstrip()` ends in one of: `:` `,` `;` `-` `–` `—` `/` `(` `[` `{`
Regex: `re.search(r"[:,;\-–—/(\[{]\s*$", text)`.

### D3 — Unbalanced opener
`True` WHEN `text.count("(") > text.count(")")` OR `text.count("[") > text.count("]")`
OR `text.count("{") > text.count("}")`.
(Catches `**Proxy Conecta.gov (` — one `(`, zero `)`.)

### D4 — Orphan markdown heading
`True` WHEN the whole `text` is a single heading-only block: a bold/heading line optionally
ending in `:`, with no non-empty body line after it.

Regex (apply to `text`, `re.DOTALL` NOT needed — body is line-based):
```python
# Strip a leading markdown heading marker, then test the single-line bold-title shape.
ORPHAN_HEADING = re.compile(
    r"""^\s*
        (?:\#{1,6}\s*)?          # optional ATX heading hashes
        \*{0,2}                   # optional opening bold
        [^\n*][^\n]*?             # title text (no newline)
        \*{0,2}                   # optional closing bold
        \s*:?                     # optional trailing colon
        \s*$                      # end — single line only
    """,
    re.VERBOSE,
)
def _is_orphan_heading(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) != 1:
        return False            # has a body line -> not orphan
    line = lines[0]
    # must look like a heading/bold title: starts with # or contains ** ... **,
    # OR ends in ':' (label-style heading)
    looks_heading = (
        line.lstrip().startswith("#")
        or ("**" in line)
        or line.rstrip().endswith(":")
    )
    return bool(looks_heading and ORPHAN_HEADING.match(line))
```
Catches: `**Causa Raiz Identificada:**`, `**Ativação OSS do Greptile:**`,
`**QueryFlow Audit Design**`, `**Memory Service (F022)**` (single bold line, no body).

### Canary table (MUST return `False` — not degenerate)
| Sentence | Why it passes |
|----------|---------------|
| `Nunca deixar commit sem push.` | last_tok=`push` ∉ CUT_STOPWORDS; no D2/D3/D4; 1 line but not heading-shaped (no `#`, no `**`, no trailing `:`) |
| `Usar sempre LIMIT em queries PostgreSQL.` | last_tok=`postgresql` ∉ list; `em` is mid-sentence, not final |
| `O token em ~/.git-credentials expira.` | last_tok=`expira` ∉ list; leading `O` and mid `em` not inspected |
| `Preferir ONNX ao invés de torch no CPU.` | last_tok=`cpu` ∉ list; `de`/`no` mid-sentence |

### Degenerate table (MUST return `True`)
| Fragment | Rule fired |
|----------|-----------|
| `Conflitos recorrentes são` | — (`são` is a verb, NOT a stopword) → see caveat below |
| `Se o fato mudou, o` | D1 (`o` final) |
| `Ao avaliar regress` | — (`regress` not a stopword) → see caveat |
| `**Causa Raiz Identificada:**` | D4 + D2 (`:`) |
| `**Proxy Conecta.gov (` | D3 (unbalanced `(`) + D2 (`(`) |
| `**Ativação OSS do Greptile:**` | D4 + D2 |
| `**QueryFlow Audit Design**` | D4 |
| `**Memory Service (F022)**` | D4 (balanced parens, single bold line) |

> **CAVEAT for rok/reg (design-open, see open questions):** `Conflitos recorrentes são`
> and `Ao avaliar regress` are NOT caught by D1 (final word is not a stopword — `são` is a
> verb, `regress` is a truncation of `regressão`). D1..D4 as specified catch 6 of the 8
> listed examples. Catching the other two needs either (a) a "no terminal punctuation AND
> short" heuristic, or (b) a known-truncation detector — both carry false-positive risk
> against the invariant. **Decision (G0):** keep the rule conservative (D1..D4) for Camada 1;
> `são`/`regress`-type truncations WITHOUT a dangling stopword are left for Camada 2 or a
> follow-up, because a safe rule for them would threaten canary #3 (`...expira.`, a verb
> ending). This is logged as an explicit limitation, not an oversight.
