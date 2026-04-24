"""Pure utilities without side effects (easy to test)."""
from __future__ import annotations

import json
import re

import json5

_CONTENT_ALIASES = ("content", "text", "markdown", "md", "body", "data", "value")


def parse_args(params) -> dict:
    """Tolerant tool-arguments parser: fixes literal newlines and
    guarantees a dict on output (the LLM sometimes sends a bare string)."""
    if isinstance(params, dict):
        return params
    if not isinstance(params, str):
        return {"content": str(params)}

    # Pre-clean: the LLM often glues tool-call wrapper tails inside arguments —
    # `{"content": "..."}</arguments`, `</tool_call>`, trailing ```, etc.
    # Without this json5 crashes, we fall back to the final path and write
    # a raw JSON blob into the file instead of content (bug was visible in draft.md).
    s = params.strip()
    for tail in ("</arguments>", "</arguments", "</tool_call>", "</tool_call", "```"):
        if s.endswith(tail):
            s = s[: -len(tail)].rstrip()

    # Double-encoded JSON: the LLM sometimes wraps args in a JSON string literal
    # instead of a JSON object — `"{\"content\": \"markdown\"}"`. If we leave it as is,
    # json5.loads parses it to a plain python string, _wrap wraps it → the file
    # gets a serialized JSON blob instead of markdown (seen in draft.md this session).
    # Unwrap up to 3 wrapping levels (never seen more in practice).
    for _ in range(3):
        if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
            try:
                inner = json.loads(s)
                if isinstance(inner, str) and inner.lstrip()[:1] in ('{', '"'):
                    s = inner.strip()
                    continue
            except Exception:
                pass
        break
    # balanced-brace trim: drop any tail after the last top-level `}`
    if s.startswith("{"):
        depth = 0
        last_close = -1
        in_str = False
        esc = False
        for i, ch in enumerate(s):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > 0:
            s = s[: last_close + 1]

    def _wrap(obj):
        if isinstance(obj, dict):
            return obj
        return {"content": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}

    try:
        return _wrap(json5.loads(s))
    except Exception:
        pass
    try:
        fixed = re.sub(
            r'"((?:[^"\\]|\\.)*)"',
            lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + '"',
            s,
            flags=re.DOTALL,
        )
        return _wrap(json5.loads(fixed))
    except Exception:
        pass
    # Last resort: manually extract the key value honoring escaped quotes.
    # `(?:\\.|[^"\\])*` eats any escaped character or non-quote/non-backslash.
    for key in ("content", "code", "query", "url"):
        m = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', s, re.DOTALL)
        if m:
            raw = m.group(1)
            # unescape standard JSON escapes
            try:
                decoded = json.loads(f'"{raw}"')
            except Exception:
                decoded = raw.replace('\\n', '\n').replace('\\"', '"').replace("\\\\", "\\")
            return {key: decoded}
    # Nothing matched. If the input LOOKS LIKE JSON (starts with `{` or
    # contains `"content"`/`"code"`) — this is a broken structure, mark with a prefix
    # so the problem shows in draft/notes. Otherwise — a bare string that the
    # LLM sent without an object: return it as content unchanged.
    looks_like_json = s.startswith("{") or re.search(r'"(?:content|code|query|url)"', s) is not None
    if looks_like_json:
        return {"content": f"[parse_args: unrecognized tool args]\n{params}"}
    return {"content": params}


def get_content(params) -> str:
    """Extracts content from tool-arguments with a soft fallback. Use in tools
    that accept markdown text (write_draft, append_notes, etc.) —
    so that KeyError does not break the whole chain when the LLM sent `{"text": ...}`,
    `{"markdown": ...}` or just one wrongly-named key.
    """

    def _unwrap(value):
        current = value
        for _ in range(4):
            if isinstance(current, dict):
                for key in _CONTENT_ALIASES:
                    if key in current:
                        current = current[key]
                        break
                else:
                    first_str = next((v for v in current.values() if isinstance(v, str)), None)
                    if first_str is None:
                        return current
                    current = first_str
                continue

            if not isinstance(current, str):
                return current

            s = current.strip()
            if not s:
                return ""

            if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
                try:
                    inner = json.loads(s)
                    if inner != current:
                        current = inner
                        continue
                except Exception:
                    pass

            looks_like_content_blob = (
                s.startswith("{")
                and any(f'"{key}"' in s for key in _CONTENT_ALIASES)
            )
            if looks_like_content_blob:
                try:
                    inner = json5.loads(s)
                except Exception:
                    try:
                        inner = json.loads(s)
                    except Exception:
                        break
                if isinstance(inner, dict):
                    current = inner
                    continue
            break
        return current

    d = parse_args(params)
    if "content" in d:
        content = _unwrap(d["content"])
        return content if isinstance(content, str) else (json.dumps(content, ensure_ascii=False) if content is not None else "")
    # Common aliases the LLM invents
    for alt in _CONTENT_ALIASES[1:]:
        if alt in d:
            content = _unwrap(d[alt])
            return content if isinstance(content, str) else (json.dumps(content, ensure_ascii=False) if content is not None else "")
    # Otherwise take the first string value — a weird draft is better than a crash.
    for v in d.values():
        if isinstance(v, str):
            content = _unwrap(v)
            return content if isinstance(content, str) else (json.dumps(content, ensure_ascii=False) if content is not None else "")
    return ""


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


# Arxiv-id: YYMM.NNNNN (YY=year, MM=01-12 month, N=4-5 digits seq).
# Strict version (2026-04): negative lookahead/behind cut off:
#   - decimal numbers in text ("effect size 2504.03 mm/s" — only 2 digits, but 2504.03456 would match)
#   - id fragments ("2301.123456" → 6 digits, not arxiv)
#   - id embedded in a longer number ("A12301.12345", "2301.12345.67")
# MM=01-12 so "2013.12345" (invalid month 13) does not match.
ARXIV_RE = re.compile(
    r"(?<![\d.])(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(?![\d.])"
)

# Stopwords for keyword-overlap checks (used by the validator).
STOPWORDS = frozenset({"paper", "papers", "article", "work", "authors", "model", "method"})


def extract_ids(text: str) -> set[str]:
    """Single source of truth for extracting arxiv-ids from any text."""
    return set(ARXIV_RE.findall(text or ""))


def keyword_set(s: str) -> set[str]:
    """Set of significant tokens for jaccard comparison.

    Includes words ≥5 letters AND 4-digit numbers (years: 2023/2024/2025…).
    Years matter for fuzzy dedup: 'stability 2023' and 'stability 2024' —
    different queries; without them jaccard=1.0 → false REJECT-duplicate loop.
    """
    words = {w.lower() for w in re.findall(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]{4,}", s)}
    years = set(re.findall(r"\b(20\d{2})\b", s))  # 2000-2099
    return words | years


# Explicit anti-keywords: if any of these shows up in abstract/description — the paper
# does not belong to the EW/ELINT/radar domain; reject without counting overlap.
# Motivation: "cognitive" is in EW_CORE, yet the same ComVo/emotional-support/LLM-safety
# paper can have 1-2 accidental overlaps with the header and pass the gate. The anti-list
# fires BEFORE positive-check and returns (False, "anti_keyword").
ANTI_KEYWORDS: frozenset[str] = frozenset({
    # audio/speech domain
    "vocoder", "text-to-speech", "tts", "mel-spectrogram", "waveform-synthesis",
    "audio-synthesis", "audio generation", "speech synthesis",
    # jailbreak / LLM safety (not ELINT-security)
    "jailbreak", "jailbreaking", "safety alignment", "llm safety",
    "harmful content", "red teaming", "prompt injection",
    # automotive / lidar
    "self-driving", "autonomous driving", "lidar", "automotive radar",
    # emotional support / dialog
    "emotional support conversation", "emotional support", "empathetic dialog",
})


def has_anti_keyword(text: str) -> str | None:
    """Returns the first anti-keyword found in the text, or None."""
    lower = text.lower()
    for kw in ANTI_KEYWORDS:
        if kw in lower:
            return kw
    return None


# Generic words from plan.md that MUST NOT serve as a domain anchor:
# they appear in any scientific abstract and yield false-positive overlap.
_TOPIC_GENERIC = frozenset({
    "modern", "approach", "approaches", "survey", "review", "study", "studies",
    "analysis", "research", "technique", "techniques", "method", "methods",
    "system", "systems", "model", "models", "application", "applications",
    "based", "using", "toward", "towards", "paper", "papers",
    "plan", "focus", "todo", "done", "blocked", "progress", "digest",
    "iter", "attempts", "evidence", "revision", "revisions",
    "deep-dive", "trade-offs", "tradeoffs",
    # generic connectors (caught "electronic" in 2508.12935 emotional support)
    "advanced", "novel", "complex", "between", "among", "within",
    "challenges", "challenge", "problem", "problems", "issue", "issues",
    "future", "recent", "current", "emerging", "integration", "integrations",
    # local noise from seeds like "ELINT fingerprinting in urban canyons"
    "canyons", "canyon", "contested", "environments", "environment",
    "algorithm", "algorithms", "architecture", "architectures",
    # false keywords from plan.md structure (headers and metadata, not domain terms)
    "vocabulary",  # from "**Core vocabulary:**" — header, not domain
})


def _plan_sections(plan_text: str) -> tuple[str, str]:
    """Splits plan.md into (header, seeds).

    header = first line (# Plan: ...) + `**Core vocabulary:** ...`,
             if LLM bootstrap produced it. This is the topic core.
    seeds  = all lines with [Tn] — they often drift and contain specific jargon,
             but on their own they are a weak anchor (see canyons / challenges).
    """
    if not plan_text:
        return "", ""
    lines = plan_text.splitlines()
    header_parts: list[str] = []
    if lines:
        header_parts.append(lines[0])
    for ln in lines[1:10]:  # vocab-line is rendered right after the header
        if ln.lstrip().lower().startswith("**core vocabulary:"):
            header_parts.append(ln)
            break
    header = "\n".join(header_parts)
    seeds = "\n".join(ln for ln in lines if re.search(r"\[T\d+\]", ln))
    return header, seeds


def extract_topic_keywords(plan_text: str) -> set[str]:
    """All domain keywords from plan.md (header ∪ seeds). Backwards-compat."""
    header, seeds = _plan_sections(plan_text)
    kws = keyword_set(header + " " + seeds)
    return {w for w in kws if w not in STOPWORDS and w not in _TOPIC_GENERIC}


def extract_topic_keywords_tiered(plan_text: str) -> tuple[set[str], set[str]]:
    """Two-tier split for the domain gate: (header_kws, seed_kws).

    header_kws — the domain core from the header (# Plan: ...). This is what the
    user actually asked for. ComVo (audio vocoder) has zero overlap with
    {electronic, warfare, elint, intelligence} — the gate cuts it.
    seed_kws — specifics from [Tn] tasks. A weak anchor (drift), used only as
    a bonus for papers that ALREADY passed by header.
    """
    header, seeds = _plan_sections(plan_text)
    h = {w for w in keyword_set(header)
         if w not in STOPWORDS and w not in _TOPIC_GENERIC}
    s = {w for w in keyword_set(seeds)
         if w not in STOPWORDS and w not in _TOPIC_GENERIC}
    return h, s - h  # seeds disjoint from header to avoid double counting


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def derive_static_vocabulary(query: str, abstracts: list[str] | None = None,
                             max_terms: int = 12) -> list[str]:
    """Derives a domain vocabulary from a query (and optionally top abstracts).

    Fallback for when LLM bootstrap crashed and core_vocabulary is empty.
    Strategy: keyword_set from query + abstracts, filter out STOPWORDS/_TOPIC_GENERIC,
    sort by frequency of occurrence in the combined text, keep top-N.

    This is a backup plan — fewer and less precise terms than the LLM produces
    (no abstraction level, no acronyms the LLM knows by context), but the
    domain gate stops relying only on 4 header words.

    Parameters:
        query: the user query
        abstracts: optional list of abstracts from the first hf_papers/arxiv hits
                   (used to extend vocabulary with terms from real papers)
        max_terms: upper bound on vocab size (matches the LLM output 8-15)

    Returns: list of terms (lowercase, deduplicated) of up to max_terms entries.
    """
    text = query
    if abstracts:
        text = text + " " + " ".join(abstracts)
    raw = keyword_set(text)
    # Filter out generic + stopwords + years
    candidates = [w for w in raw
                  if w not in STOPWORDS
                  and w not in _TOPIC_GENERIC
                  and not re.fullmatch(r"20\d{2}", w)
                  and len(w) >= 4]
    if not candidates:
        return []
    # Sort by frequency of occurrence in text (the more frequent, the more characteristic).
    lower = text.lower()
    scored = [(w, lower.count(w)) for w in candidates]
    scored.sort(key=lambda p: (-p[1], p[0]))
    return [w for w, _ in scored[:max_terms]]
