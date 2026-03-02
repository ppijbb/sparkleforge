"""SmartSnippetExtractor — windows around FTS5 match positions.

Returns relevant snippets instead of dumb truncation.
Ported from reference/claude-context-mode server.ts.
"""

STX = "\x02"
ETX = "\x03"


def positions_from_highlight(highlighted: str) -> list[int]:
    """Parse FTS5 highlight markers; return character offsets of matches in clean text."""
    positions: list[int] = []
    clean_offset = 0
    i = 0
    while i < len(highlighted):
        if highlighted[i] == STX:
            positions.append(clean_offset)
            i += 1
            while i < len(highlighted) and highlighted[i] != ETX:
                clean_offset += 1
                i += 1
            if i < len(highlighted):
                i += 1
        else:
            clean_offset += 1
            i += 1
    return positions


def extract_snippet(
    content: str,
    query: str,
    max_len: int = 1500,
    highlighted: str | None = None,
) -> str:
    """Extract windows around match positions. Fallback: indexOf on query terms."""
    if len(content) <= max_len:
        return content
    positions: list[int] = []
    if highlighted:
        positions = positions_from_highlight(highlighted)
    if not positions:
        terms = [t for t in query.lower().split() if len(t) > 2]
        lower = content.lower()
        for term in terms:
            idx = lower.find(term)
            while idx != -1:
                positions.append(idx)
                idx = lower.find(term, idx + 1)
    if not positions:
        return content[:max_len] + "\n…"
    positions.sort()
    window = 300
    windows: list[tuple[int, int]] = []
    for pos in positions:
        start = max(0, pos - window)
        end = min(len(content), pos + window)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    parts: list[str] = []
    total = 0
    for start, end in windows:
        if total >= max_len:
            break
        part_len = min(max_len - total, end - start)
        part = content[start : start + part_len]
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(content) else ""
        parts.append(prefix + part + suffix)
        total += len(part)
    return "\n\n".join(parts)
