import copy
import os

import pytest

from tgraph import visualize as V


@pytest.mark.parametrize("backend", ["dot", "mpl"])
def test_render_backends(records, tmp_path, backend):
    if backend == "dot" and not V.has_dot():
        pytest.skip("no dot binary")
    g = next(r for r in records if r["tier"] == "L4")["target_graph"]
    out = V.render(g, str(tmp_path / "g"), fmt="png", backend=backend)
    assert os.path.getsize(out) > 1000


def test_diff_pair_neurons(records, tmp_path):
    g = next(r for r in records if r["tier"] == "L2")["target_graph"]
    p = copy.deepcopy(g)
    p["edges"].pop(2)
    p["nodes"][3]["kind"] = "GELU"
    _, st, est = V.diff_graph(p, g)
    assert "missing" in est.values()
    for fn in (lambda: V.render_diff(p, g, str(tmp_path / "d"), fmt="svg"),
               lambda: V.render_pair(p, g, str(tmp_path / "p"), fmt="svg")):
        assert os.path.getsize(fn()) > 1000
    l0 = next(r for r in records if r["tier"] == "L0")["target_graph"]
    assert os.path.getsize(V.render(l0, str(tmp_path / "n"), fmt="svg", expand_neurons=True)) > 500
