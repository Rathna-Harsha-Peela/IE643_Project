import json

import pytest
import torch

from tgraph.oracle import extract_reference_graph, safe_reference_graph


def test_oracle_reproduces_all_5000(records):
    bad = [r["sample_id"] for r in records
           if json.dumps(extract_reference_graph(r["source_code"]), sort_keys=True)
           != json.dumps(r["target_graph"], sort_keys=True)]
    assert len(records) == 5000
    assert bad == [], bad[:10]


NESTED = """
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)) + x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.b1 = Block(8)
    def forward(self, x):
        return self.b1(self.stem(x))
"""


def test_nested_scope():
    g = extract_reference_graph(NESTED)
    conv = next(n for n in g["nodes"] if n["name"] == "b1_conv")
    assert conv["scope"] == "b1.conv" and conv["kind"] == "Conv2d" and conv["shape"] == [1, 8, 16, 16]


UNTRACEABLE = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        if x.sum() > 0:
            return self.fc(x)
        return x
"""


def test_data_dependent_if_raises():
    with pytest.raises(torch.fx.proxy.TraceError):
        extract_reference_graph(UNTRACEABLE, input_shape=(1, 4))


def test_safe_oracle_reports_errors():
    g, err = safe_reference_graph(UNTRACEABLE, input_shape=(1, 4))
    assert g is None and err.startswith("not traceable: TraceError")
    g, err = safe_reference_graph("def broken(:\n  pass")
    assert g is None and "SyntaxError" in err
    g, err = safe_reference_graph(NESTED)
    assert err is None and g["nodes"][-1]["op"] == "output"


TWO_INPUT = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(6, 8)
    def forward(self, x, y):
        return self.a(x) + self.b(y)
"""


def test_two_inputs_keep_arg_names():
    g = extract_reference_graph(TWO_INPUT, input_shape=[(1, 4), (1, 6)])
    assert [n["name"] for n in g["nodes"] if n["op"] == "placeholder"] == ["x", "y"]
