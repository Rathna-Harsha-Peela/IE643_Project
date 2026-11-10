"""Calls the app's core handler directly (no UI)."""
import os

import pytest

from app import core
from tgraph.io import PROC_DIR

GOOD = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 16 * 16, 2)
    def forward(self, x):
        return self.fc(torch.flatten(torch.relu(self.conv(x)), 1))
"""


def test_oracle_only_renders():
    r = core.handle(GOOD, "Oracle only")
    assert r["error"] is None and r["ref_img"] and os.path.exists(r["ref_img"])


def test_bad_inputs_are_friendly():
    assert "syntax" in core.handle("class A(:\n", "Oracle only")["error"].lower()
    assert "nn.Module" in core.handle("x = 1", "Oracle only")["error"]
    assert core.handle("", "Oracle only")["error"]
    assert "shape" in core.handle(GOOD, "Oracle only", shape_override="a,b")["error"].lower()


def test_untraceable_message():
    src = open(os.path.join(os.path.dirname(PROC_DIR), "ood", "untraceable", "data_dependent_if.py")).read()
    r = core.handle(src, "Oracle only")
    assert r["ref_img"] is None and "not traceable" in r["ref_msg"]


def test_precomputed_examples_render_all_tabs():
    pre = core.precomputed()
    if not pre:
        pytest.skip("no stored predictions yet")
    cfg = next(iter(pre))
    for label, code in core.examples():
        r = core.handle(code, f"Precomputed ({cfg})")
        assert r["error"] is None or "could not be parsed" in r["error"], (label, r["error"])
        if label.startswith("UNT_"):
            assert "not traceable" in r["ref_msg"]
