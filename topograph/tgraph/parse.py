"""Robust extraction of CG-JSON from raw LLM text.

parse_output(text) -> (cg | None, info) where info = {parse_ok, repaired, schema_ok, coercions, error}.
"""
import json
import re

import jsonschema

from .schema import SCHEMA

_THINK = re.compile(r"<think>.*?</think>", re.S)
_VALIDATOR = jsonschema.Draft7Validator(SCHEMA)


def _strip(text):
    text = _THINK.sub("", text)
    if "</think>" in text:                       # template pre-filled "<think>" so only the close tag is in output
        text = text.split("</think>", 1)[1]
    return text.replace("```json", "").replace("```", "")


def _first_balanced_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]                          # unbalanced (truncated): let json_repair try


def _coerce(obj):
    """Numeric strings -> ints in shapes/edges; returns number of coercions."""
    n = 0
    if not isinstance(obj, dict):
        return 0
    for node in obj.get("nodes", []) if isinstance(obj.get("nodes"), list) else []:
        if isinstance(node, dict) and isinstance(node.get("shape"), list):
            for k, v in enumerate(node["shape"]):
                if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                    node["shape"][k] = int(v)
                    n += 1
        if isinstance(node, dict) and node.get("shape") is None and "shape" in node:
            node["shape"] = []                   # tolerated: non-tensor outputs
            n += 1
    for e in obj.get("edges", []) if isinstance(obj.get("edges"), list) else []:
        if isinstance(e, list):
            for k, v in enumerate(e):
                if isinstance(v, str) and v.strip().isdigit():
                    e[k] = int(v)
                    n += 1
    return n


def parse_output(text):
    info = {"parse_ok": False, "repaired": False, "schema_ok": False, "coercions": 0, "error": None}
    body = _first_balanced_object(_strip(text or ""))
    if body is None:
        info["error"] = "no JSON object found"
        return None, info
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        try:
            import json_repair
            obj = json_repair.loads(body)
            info["repaired"] = True
        except Exception as e:  # noqa: BLE001
            info["error"] = f"repair failed: {e}"
            return None, info
    if not isinstance(obj, dict) or not obj:
        info["error"] = "not a JSON object"
        return None, info
    info["parse_ok"] = True
    info["coercions"] = _coerce(obj)
    errs = sorted(_VALIDATOR.iter_errors(obj), key=lambda e: list(e.path))
    if errs:
        info["error"] = f"schema: {errs[0].message[:200]}"
        # keep a usable graph if the structure is recoverable
        if isinstance(obj.get("nodes"), list) and isinstance(obj.get("edges"), list):
            ok_nodes = all(isinstance(n, dict) and all(k in n for k in ("name", "op", "kind")) for n in obj["nodes"])
            ok_edges = all(isinstance(e, list) and len(e) == 2 and all(isinstance(v, int) for v in e)
                           for e in obj["edges"])
            if ok_nodes and ok_edges:
                obj = {"nodes": [{"name": str(n["name"]), "op": str(n["op"]), "kind": str(n["kind"]),
                                  "shape": n.get("shape") if isinstance(n.get("shape"), list) else []}
                                 for n in obj["nodes"]], "edges": obj["edges"]}
                return obj, info
        return None, info
    info["schema_ok"] = True
    return obj, info
