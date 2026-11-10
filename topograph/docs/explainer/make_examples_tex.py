"""Writes examples.tex (the worked-examples section) from examples.json + commentary.   python make_examples_tex.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(HERE, "examples.json")))

# which images to show per example: (file key or literal file, caption)
LAYOUT = {
    "ex01": [("ref", "Reference"), ("base9b_zs_c:diff", "9B zero-shot vs reference"), ("ft9b_c:pred", "9B fine-tuned (exact)")],
    "ex02": [("ref", "Reference (dead code removed)"), ("ft9b_c:pred", "9B fine-tuned (exact)")],
    "ex03": [("ref", "Reference"), ("base9b_fs_c:diff", "9B 3-shot vs reference")],
    "ex04": [("ref", "Reference"), ("base9b_fs_c:pred", "9B 3-shot (exact)")],
    "ex05": [("ref", "Reference"), ("ft2b_c:diff", "2B fine-tuned vs reference")],
    "ex06": [("ref", "Reference")],
    "ex07": [("ref", "Reference"), ("ft9b_c:diff", "9B fine-tuned vs reference"), ("ft9b_aug_c:pred", "9B FT + synthetic (exact)")],
    "ex08": [("ref", "Reference"), ("ft9b_c:diff", "9B fine-tuned vs reference"), ("ft9b_aug_c:pred", "9B FT + synthetic (exact)")],
    "ex09": [("ref", "Reference"), ("ft9b_c_retry:initial_diff", "first attempt vs reference"), ("ft9b_c_retry:diff", "after 1 retry round")],
    "ex10": [("ref", "Reference"), ("ft9b_aug_c:diff", "9B FT + synthetic vs reference")],
    "ex11": [("ref", "Reference"), ("ft9b_aug_c:pred", "9B FT + synthetic (exact)"), ("ceil_fs_c:diff", "27B 3-shot vs reference")],
    "ex12": [("ft9b_c:pred", "9B fine-tuned prediction (no reference exists)")],
}

TEXT = {
    "ex01": r"""\textbf{What to look at.} Every hidden unit is its own \texttt{Linear(k,1)} (``neurons'' made explicit). This
sample is an \emph{augmented} variant: the forward argument is renamed (\texttt{inp\_tensor}), a dead layer is inserted
and the \texttt{\_\_init\_\_} order is shuffled. The zero-shot 9B model wires the graph correctly (it is
\emph{isomorphic} to the reference) but names the input after the argument instead of \texttt{x}, picks the wrong input
width ($[1,3]$ instead of $[1,2]$) and therefore gets node and edge \emph{names} wrong: strict edge F1 is only 0.55
although the shape of the graph is right. The fine-tuned 9B reproduces the reference exactly.""",
    "ex02": r"""\textbf{What to look at.} \texttt{self.unused\_op\_*} is declared and even \emph{called}, but its result never
reaches the \texttt{return}, so \texttt{torch.fx} (after dead-code elimination) does not contain it. The validator
checks exactly this (rule ``dead code included''). The fine-tuned model skips the dead call correctly. Over the whole test
set the base Coder-7B model included dead code in 34.6\% of the dead-code samples, the 9B 3-shot model in 0\%.""",
    "ex03": r"""\textbf{What to look at.} A residual network whose structure was \emph{held out} from fine-tuning. The 3-shot
base model gets every node right (node F1 1.0) but miswires the edges: the red dashed edges are reference edges it
missed (including the long \emph{skip connection} into an \texttt{add}), orange edges are ones it invented, and it emits
edges to node indices that do not exist (\texttt{edge [17, 18] points outside the 18 nodes}). Long-range skip edges
are the single most common failure of the prompted models (the \texttt{missing\_skip\_edge} label in Figure~\ref{fig:F7}).
The fine-tuned 9B gets this held-out topology exactly right.""",
    "ex04": r"""\textbf{What to look at.} Several parallel branches read the same tensor and are joined by
\texttt{torch.cat}. Branching and merging nodes are easy to spot in the drawing (out-degree $>1$, in-degree $>1$).
Here even the prompted 3-shot model is exactly right; tier L3 is the easiest multi-path tier for it (60.5\% exact).""",
    "ex05": r"""\textbf{What to look at.} The nested L4 pattern (stem $\to$ two paths $\to$ concat $\to$ merge $\to$
residual add $\to$ head) with the one L4 structure that was held out of training. The fine-tuned \textbf{2B} writes
the nodes correctly but wires them like the L4 templates it saw in training: the validator reports
\texttt{'cat' has 1 input}, \texttt{'merge' has 2 inputs; a Conv2d takes exactly 1} and \texttt{'add' has 1 input
(missing skip edge?)}. All 52 held-out L4 test samples fail this way for the 2B (0\%), while the fine-tuned
\textbf{9B} gets 100\% of them. This is the clearest memorisation-vs-generalisation contrast in the project.""",
    "ex06": r"""\textbf{What to look at.} Nothing is wrong with the code; the failure is in decoding. The zero-shot 9B starts
a plausible answer, then falls into a loop in the edge list (\texttt{[n,n+1]} with ever larger \texttt{n}) until the
3{,}072-token limit stops it. The output is not a usable graph (parse/schema failure). 22\% of zero-shot test answers
ended like this (mostly on the larger L2/L4 graphs); fine-tuned models never did (Section~\ref{sec:journey}).""",
    "ex07": r"""\textbf{What to look at.} An out-of-distribution module: a \emph{submodule class} \texttt{ResUnit} used twice.
\texttt{torch.fx} inlines nested modules and names their layers with the full path (\texttt{unit\_a\_conv\_a},
scope \texttt{unit\_a.conv\_a}). The dataset only has flat models, so the fine-tuned 9B writes \texttt{conv\_a} (a
name that is not declared at the top level: the validator flags it as possibly hallucinated) and collapses the two
units into one (the whole second unit is red dashed = missing). After adding synthetic training modules that contain
nested classes, the model gets it exactly right.""",
    "ex08": r"""\textbf{What to look at.} \texttt{nn.ModuleList} plus a Python \texttt{for} loop: \texttt{torch.fx}
unrolls the loop and names the layers \texttt{layers\_0}, \texttt{layers\_1}, \texttt{layers\_2}. The fine-tuned 9B
draws the right structure (isomorphic) but names the nodes \texttt{layers}, \texttt{layers\_1}, \dots, so strict
metrics and the code-consistency check fail (``naming only'' failure). The synthetic-data model names them correctly.""",
    "ex09": r"""\textbf{What to look at.} \emph{Verify-and-retry}: the validator's messages (which never use the reference)
are sent back to the model as a new chat turn. The first attempt left the output node disconnected
(\texttt{output node needs >=1 input}, \texttt{node 0 'x' does not lead to the output}). After one retry round the graph
passes every validity check, but it is \emph{still not exactly right} (edge F1 0.87, one node missing): a valid graph
is not necessarily a correct graph. Over OOD-1, retry raised the fine-tuned 9B from 47.5\% to 67.5\% valid and from
30\% to 45\% exact.""",
    "ex10": r"""\textbf{What to look at.} OOD-2 deliberately uses operators that appear in neither the dataset nor the
synthetic generator, here \texttt{nn.ConvTranspose2d} (an up-sampling convolution). The synthetic-data 9B gets the
\emph{topology} exactly right (all edges black) but almost all \emph{shapes} wrong (red shape text, shape accuracy
11\%). The mistake starts at the very first layer: the model ignores the \texttt{stride=2} of the ordinary
convolution \texttt{down} and predicts \texttt{[1,8,16,16]} instead of \texttt{[1,8,8,8]}. Every later shape inherits
the error, so the transposed convolution then ``doubles'' 16 to 32 instead of 8 to 16. The shape check catches the
inconsistency at the final skip addition (\texttt{add\_1 inputs [1,4,16,16] and [1,4,32,32] are not
broadcast-compatible}). The 27B model gets
this one fully right. Shape propagation through unseen operators is the main remaining weakness of all fine-tuned
models.""",
    "ex11": r"""\textbf{What to look at.} A tiny attention block (\texttt{unsqueeze}, \texttt{matmul}, \texttt{softmax},
\texttt{squeeze}; method calls become \texttt{call\_method} nodes). Here the roles flip: the synthetic-data 9B is
exactly right, while the 27B gets the topology right but adds an edge to a node index that does not exist and gets 31\%
of the shapes wrong, so the validator rejects it. Bigger is better on average, not on every example.""",
    "ex12": r"""\textbf{What to look at.} The \texttt{if} depends on the \emph{value} of the input, so
\texttt{torch.fx.symbolic\_trace} raises \texttt{TraceError}: there is no reference graph, and this is exactly where an
LLM can still be useful. The fine-tuned 9B draws both branches feeding the output (it cannot know which branch runs),
which the validator reports as \texttt{output node has 2 inputs}. The validator also wrongly reports \texttt{conv} as
dead code: its code analysis only follows straight-line code and does not look inside \texttt{if} bodies (a known
limitation, Section~\ref{sec:caveats}). These examples are qualitative only.""",
}


def esc(s):
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "{": r"\{", "}": r"\}",
           "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
    return "".join(rep.get(c, c) for c in s)


def img(e, key):
    if key == "ref":
        return e.get("ref_png")
    cfg, what = key.split(":")
    p = next(p for p in e["preds"] if p["config"] == cfg)
    return p.get({"diff": "diff_png", "pred": "pred_png", "initial_diff": "initial_diff_png"}[what])


def main():
    out = []
    for e in EX:
        code_file = f"code/{e['id']}.py"
        with open(os.path.join(HERE, code_file), "w") as f:
            f.write(e["code"])
        where = {"test": "official test", "ood": "OOD-1", "ood2": "OOD-2", "untraceable": "untraceable set"}[e["eval_set"]]
        meta = [f"set: {where}", f"sample: \\texttt{{{esc(e['sample_id'])}}}"]
        if e.get("tier") and e["tier"].startswith("L"):
            meta.append(f"tier {e['tier']}")
        if e.get("aug") and e["aug"] != "none":
            meta.append(f"augmentation: \\texttt{{{esc(e['aug'])}}}")
        if e["eval_set"] == "test":
            meta.append("topology seen in training" if e.get("struct_seen") else "topology \\textbf{held out} from training")
        if e.get("category"):
            meta.append(f"category: \\texttt{{{esc(e['category'])}}}")
        if e.get("ref_nodes"):
            meta.append(f"{e['ref_nodes']} nodes")
        out.append(f"\\subsection{{Example {int(e['id'][2:])}: {e['title']}}}\\label{{sec:{e['id']}}}")
        out.append("{\\small " + " \\textbullet{} ".join(meta) + "}\n")
        out.append(f"\\lstinputlisting[style=py]{{{code_file}}}")
        rows = []
        for p in e["preds"]:
            m = p.get("metrics") or {}
            fl = ", ".join(x.replace("_", " ") for x in (p.get("failures") or [])) or ("--" if m else "n/a")
            rows.append(f"{p['label']} & {'yes' if p['valid'] else 'no'} & "
                        f"{m.get('edge_f1', float('nan')):.2f} & {'yes' if m.get('exact_strict') else 'no' if m else 'n/a'} & "
                        f"{m.get('shape_acc', float('nan')):.2f} & {p['n_tokens']} & \\raggedright {esc(fl)} \\tabularnewline"
                        .replace("nan", "--"))
        out.append("\\begin{center}\\footnotesize\\begin{tabular}{lcccccp{4.6cm}}\\toprule")
        out.append("model & valid & edge F1 & exact & shape acc. & tokens & failure labels \\\\ \\midrule")
        out += rows
        out.append("\\bottomrule\\end{tabular}\\end{center}")
        out.append(TEXT[e["id"]] + "\n")
        lay = LAYOUT[e["id"]]
        n = len(lay)
        h = {1: "0.36", 2: "0.40", 3: "0.40"}[n]
        w = {1: "0.55", 2: "0.47", 3: "0.32"}[n]
        out.append("\\begin{figure}[H]\\centering")
        for key, cap in lay:
            f = img(e, key)
            if not f:
                continue
            out.append(f"\\begin{{minipage}}[b]{{{w}\\linewidth}}\\centering"
                       f"\\includegraphics[width=\\linewidth,height={h}\\textheight,keepaspectratio]{{figs/{f}}}\\\\"
                       f"\\footnotesize {cap}\\end{{minipage}}\\hfill")
        out.append(f"\\caption{{Example {int(e['id'][2:])} ({esc(e['sample_id'])}). Diff colours: black = matched, red dashed = "
                   f"missing (reference only), orange = hallucinated (prediction only), red text = wrong kind or shape.}}")
        out.append("\\end{figure}")
        if e["kind"] == "loop":
            p = e["preds"][0]
            out.append("\\noindent\\textit{End of the zero-shot answer (cut off at 3{,}072 tokens):}")
            out.append("\\begin{lstlisting}[style=plain]\n..." + p["raw_tail"].replace("\n", " ").replace(",", ", ")[-230:] + "\n\\end{lstlisting}")
        out.append("\\clearpage\n")
    with open(os.path.join(HERE, "examples.tex"), "w") as f:
        f.write("\n".join(out))
    print("wrote examples.tex")


if __name__ == "__main__":
    main()
