"""generate.run_set with a fake backend: few-shot rendering, verify-and-retry, self-consistency (no model needed)."""
import copy
import json
import os

import pytest

from scripts.generate import few_shot_exemplars, load_config, run_set
from tgraph.io import load_eval_set
from tgraph.schema import canonical_to_cg

MODEL = os.path.join(os.environ.get("MODELS", ""), "Qwen3.5-0.8B")


class FakeBackend:
    """First call returns graphs with a dropped edge; later calls (retry turns) return the gold graph."""

    def __init__(self, gold):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.gold, self.calls, self.prompts = gold, 0, []

    def generate(self, prompts, cfg, constrained, n=1):
        self.calls += 1
        self.prompts.append(prompts)
        out = []
        for p in prompts:
            sid = next(s for s in self.gold if self.gold[s]["marker"] in p.rsplit("Source code:", 1)[1])
            cg = copy.deepcopy(self.gold[sid]["cg"])
            if self.calls == 1:
                cg["edges"].pop(0)
            out.append([{"text": json.dumps(cg), "n_output_tokens": 10, "finish_reason": "stop",
                         "n_prompt_tokens": 100, "latency_s": 0.0} for _ in range(n)])
        return out


@pytest.mark.skipif(not os.path.isdir(MODEL), reason="tokenizer not available")
def test_fewshot_retry_and_sc():
    rows = load_eval_set("test_mini")[:3]
    gold = {r["sample_id"]: {"cg": canonical_to_cg(r["target_graph"]), "marker": r["source_code"][:80]} for r in rows}
    be = FakeBackend(gold)
    cfg = dict(load_config("base9b_fs_c_retry"))
    out = run_set(be, cfg, rows, few_shot_exemplars())
    assert be.calls == 2                                   # initial + one retry round (all fixed after it)
    assert all(not o["initial_fully_valid"] and o["validation"]["fully_valid"] and o["rounds_used"] == 1 for o in out)
    first = be.prompts[0][0]
    assert first.count("<|im_start|>assistant") == 4       # 3 few-shot answers + generation prompt
    retry = be.prompts[1][0]
    assert "Your graph has these problems:" in retry and retry.endswith("<think>\n\n</think>\n\n")
    be2 = FakeBackend(gold)
    cfg2 = dict(load_config("ft9b_sc5"))
    out2 = run_set(be2, cfg2, rows, None)
    assert be2.calls == 1 and all(o["n_output_tokens"] == 50 for o in out2)
