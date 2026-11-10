"""System prompt, user template, chat rendering and few-shot exemplars.

The SAME rendered string is used for training (scripts/prepare_data.py) and inference (scripts/generate.py).
"""
from .schema import canonical_to_cg, dumps_cg

SYSTEM_PROMPT = """You convert PyTorch nn.Module source code into the forward-pass dataflow graph that
torch.fx.symbolic_trace would produce after dead-code elimination.

Rules:
1. Nodes are listed in execution order: one input node {"op":"placeholder","kind":"placeholder"} per forward()
   input, then one node per executed operation, then exactly one final node
   {"name":"output","op":"output","kind":"output"}. If forward() has a single input, name its placeholder "x"
   regardless of the argument name; with several inputs, use the argument names.
2. self.<attr>(...) on a submodule -> op "call_module", name = attribute path with "." replaced by "_",
   kind = the module class name (e.g. "Conv2d", "BatchNorm2d", "ReLU", "Linear").
3. torch functions and torch.nn.functional calls (torch.relu, torch.cat, torch.flatten, F.relu, ...) and
   Python operators on tensors (a + b -> "add", a * b -> "mul") -> op "call_function",
   kind = the function name ("relu", "cat", "add", ...).
4. Tensor methods (x.view(...), x.flatten(1), x.mean(...)) -> op "call_method", kind = the method name.
5. Node names are globally unique: the first use of a name is plain ("relu"), later ones get suffixes
   "relu_1", "relu_2", ... in execution order (this also applies when a module attribute is called twice).
6. Skip anything that does not contribute to the returned value (dead code). Plain variable aliasing
   (y = x) creates no node.
7. "shape" is the output tensor shape assuming batch size 1. Conv networks receive 16x16 spatial inputs,
   i.e. [1, C_in, 16, 16]; networks whose first layer is Linear receive [1, in_features]. With several
   inputs, each input's shape follows the same rule using the first layer that consumes it.
8. "edges" is a list of [i, j] pairs (indices into "nodes"): the output of node i is a direct input of
   node j, listed in the order node j consumes them. The output node consumes the returned tensor.

Return only the JSON object {"nodes": [...], "edges": [...]} on a single line (compact, no indentation) with no
explanation."""

USER_TEMPLATE = "Source code:\n```python\n{code}\n```"

RETRY_TEMPLATE = "Your graph has these problems:\n{errors}\nReturn the corrected full JSON graph."


def user_message(code, input_shape_hint=None):
    msg = USER_TEMPLATE.format(code=code)
    if input_shape_hint:
        msg += f"\nInput shape: {input_shape_hint}"
    return msg


def build_messages(code, exemplars=None, input_shape_hint=None):
    """exemplars: list of (source_code, target_graph_canonical) used as few-shot user/assistant turns."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, tg in exemplars or []:
        msgs.append({"role": "user", "content": user_message(src)})
        msgs.append({"role": "assistant", "content": dumps_cg(canonical_to_cg(tg))})
    msgs.append({"role": "user", "content": user_message(code, input_shape_hint)})
    return msgs


def render(tokenizer, messages, thinking=False, add_generation_prompt=True):
    """Chat-template rendering. Qwen3.5 gets enable_thinking; other families ignore the kwarg if unsupported."""
    kw = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kw)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kw)


def turn_end_suffix(tokenizer):
    """Text the chat template appends after an assistant message's content (expected '<|im_end|>' + newline)."""
    marker = "@@CONTENT@@"
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": marker}]
    full = render(tokenizer, msgs, add_generation_prompt=False)
    return full[full.rindex(marker) + len(marker):]
