import json

from tgraph.parse import parse_output
from tgraph.schema import canonical_to_cg, cg_to_canonical, dumps_cg


def test_roundtrip_all(records):
    for r in records:
        g = r["target_graph"]
        assert cg_to_canonical(canonical_to_cg(g)) == g, r["sample_id"]


def test_parse_variants(records):
    cg = canonical_to_cg(records[0]["target_graph"])
    s = dumps_cg(cg)
    for text in [s, f"```json\n{s}\n```", f"<think>\nhmm {{}}\n</think>\n{s}", f"Here it is: {s} done",
                 f"</think>{s}"]:
        got, info = parse_output(text)
        assert got == cg and info["schema_ok"] and not info["repaired"], text[:40]
    got, info = parse_output(s[:-2])            # truncated -> repaired
    assert info["parse_ok"] and info["repaired"]
    got, info = parse_output("no json here")
    assert got is None and not info["parse_ok"]
    bad = json.loads(s)
    bad["nodes"][1]["shape"] = ["1", "1"]
    got, info = parse_output(json.dumps(bad))
    assert info["coercions"] == 2 and got["nodes"][1]["shape"] == [1, 1]
