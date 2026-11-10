import copy
import re

import pytest

from tgraph.schema import canonical_to_cg
from tgraph.validate import validate


def test_gold_graphs_have_zero_errors(records):
    bad = {}
    for r in records:
        rep = validate(canonical_to_cg(r["target_graph"]), r["source_code"])
        if rep.errors or not rep.fully_valid:
            bad[r["sample_id"]] = [e["msg"] for e in rep.errors][:3]
    assert not bad, list(bad.items())[:5]


def _codes(cg, src):
    return {e["code"] for e in validate(cg, src).errors}


@pytest.fixture(scope="module")
def residual(records):
    r = next(x for x in records if x["tier"] == "L2" and x["augmentation_type"] == "aug_variant_2")
    return r, canonical_to_cg(r["target_graph"])


def test_delete_skip_edge(residual):
    r, cg = residual
    add = next(i for i, n in enumerate(cg["nodes"]) if n["kind"] == "add")
    srcs = [a for a, b in cg["edges"] if b == add]
    skip = min(srcs)                                  # the earlier input is the skip connection
    c = copy.deepcopy(cg)
    c["edges"].remove([skip, add])
    assert _codes(c, r["source_code"]) & {"DEAD_END", "ARITY"}


def test_back_edge_cycle(residual):
    r, cg = residual
    c = copy.deepcopy(cg)
    c["edges"].append([len(c["nodes"]) - 2, 1])
    assert "CYCLE" in _codes(c, r["source_code"])


def test_undeclared_module(residual):
    r, cg = residual
    c = copy.deepcopy(cg)
    i = next(i for i, n in enumerate(c["nodes"]) if n["op"] == "call_module")
    c["nodes"][i]["name"] = "totally_new_layer"
    assert "UNDECLARED_MODULE" in _codes(c, r["source_code"])


def test_dead_code_inserted(residual):
    r, cg = residual
    dead = re.search(r"self\.(unused_op_\d+)", r["source_code"]).group(1)
    c = copy.deepcopy(cg)
    out = len(c["nodes"]) - 1
    feeder = next(a for a, b in c["edges"] if b == out)
    c["nodes"].insert(out, {"name": dead, "op": "call_module", "kind": "Identity",
                            "shape": c["nodes"][feeder]["shape"]})
    c["edges"] = [[a, b + 1 if b == out else b] for a, b in c["edges"]] + [[feeder, out]]
    codes = _codes(c, r["source_code"])
    assert "DEAD_CODE_INCLUDED" in codes and "DEAD_END" in codes


def test_conv_shape_change(residual):
    r, cg = residual
    c = copy.deepcopy(cg)
    i = next(i for i, n in enumerate(c["nodes"]) if n["kind"] == "Conv2d")
    c["nodes"][i]["shape"][1] += 5
    assert "SHAPE" in _codes(c, r["source_code"])


def test_drop_output(residual):
    r, cg = residual
    c = copy.deepcopy(cg)
    out = len(c["nodes"]) - 1
    c["nodes"].pop()
    c["edges"] = [e for e in c["edges"] if out not in e]
    assert "OUTPUT_COUNT" in _codes(c, r["source_code"])


def test_missing_module_and_parse_fail(residual):
    r, cg = residual
    c = copy.deepcopy(cg)
    i = next(i for i, n in enumerate(c["nodes"]) if n["kind"] == "BatchNorm2d")
    a = next(a for a, b in c["edges"] if b == i)
    succ = [b for x, b in c["edges"] if x == i]
    c["edges"] = [e for e in c["edges"] if i not in e] + [[a, s] for s in succ]
    del c["nodes"][i]
    c["edges"] = [[x - (x > i), y - (y > i)] for x, y in c["edges"]]
    assert "MISSING_MODULE" in _codes(c, r["source_code"])
    rep = validate(None, r["source_code"], parse_info={"parse_ok": False, "error": "bad"})
    assert not rep.fully_valid and rep.levels_passed["V0"] is False
