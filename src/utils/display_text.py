from __future__ import annotations

import ast
import json
import re
from typing import Any


def humanize_display_text(value: Any, *, bullet_keys: bool = True) -> str:
    """
    Turn dict / JSON-ish / repr(dict) strings into readable prose for UI.
    Lists become newline-separated bullets; plain strings pass through cleaned.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return _dict_to_readable(value, bullet_keys=bullet_keys)
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            line = humanize_display_text(item, bullet_keys=bullet_keys)
            if line:
                lines.append(line if line.startswith("- ") else f"- {line}")
        return "\n".join(lines)
    s = str(value).strip()
    if not s:
        return ""
    parsed = _try_parse_dictish(s)
    if isinstance(parsed, dict):
        return _dict_to_readable(parsed, bullet_keys=bullet_keys)
    if isinstance(parsed, list):
        return humanize_display_text(parsed, bullet_keys=bullet_keys)
    return _strip_code_fences(s)


def humanize_single_line(value: Any, *, max_len: int = 220) -> str:
    """One line for tables / captions; flattens dicts without newlines."""
    text = humanize_display_text(value, bullet_keys=True)
    one = " ".join(text.split())
    if len(one) > max_len:
        one = one[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return one


def _strip_code_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _try_parse_dictish(s: str) -> Any | None:
    t = s.strip()
    if not t:
        return None
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(t)
        except (ValueError, SyntaxError):
            pass
    return None


def _pretty_key(key: str) -> str:
    k = str(key).strip()
    if not k:
        return ""
    k = k.replace("_", " ")
    return k[:1].upper() + k[1:] if k else k


def _dict_to_readable(d: dict[Any, Any], *, bullet_keys: bool) -> str:
    parts: list[str] = []
    for key, val in d.items():
        label = _pretty_key(str(key))
        inner = humanize_display_text(val, bullet_keys=bullet_keys)
        inner_one = " ".join(inner.split()) if inner else ""
        if not label:
            parts.append(inner_one)
        elif bullet_keys and inner_one:
            parts.append(f"- {label}: {inner_one}")
        elif inner_one:
            parts.append(f"{label}: {inner_one}")
        elif label:
            parts.append(f"- {label}" if bullet_keys else label)
    return "\n".join(parts) if parts else ""


def normalize_loose_json_string_field(raw: Any) -> str:
    """Coerce any JSON-ish field from the model into a single display string."""
    return humanize_display_text(raw, bullet_keys=True)


def coerce_dict(value: Any) -> dict[str, Any] | None:
    """If value is a dict or a string encoding a dict, return a str-keyed dict."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        p = _try_parse_dictish(value.strip())
        if isinstance(p, dict):
            return {str(k): v for k, v in p.items()}
    return None
