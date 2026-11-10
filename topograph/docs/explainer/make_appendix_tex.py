"""Writes appendix.tex for the explainer: every detail behind the main text, generated from the result files.

    python docs/explainer/make_appendix_tex.py

Sources: results/tables/*.csv, docs/explainer/data/{gen_stats.csv,compute_log.psv}, configs/*.yaml,
data/processed/*, the package's own code (prompts, schema, docstrings). Nothing is typed in by hand except prose.
"""
import ast
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import yaml
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from tgraph import prompts as P  # noqa: E402

T = os.path.join(REPO, "results", "tables")
PROC = os.path.join(REPO, "data", "processed")

LABEL = {
    "base2b_fs_c": "2B 3-shot", "base9b_zs_c": "9B zero-shot", "base9b_fs_c": "9B 3-shot",
    "base9b_fs_u": "9B 3-shot, unconstrained", "base9b_fs_c_retry": "9B 3-shot + retry",
    "base9b_think": "9B thinking (8k)", "base9b_think32k": "9B thinking (30k)",
    "base9b_think32k_c": "9B think-then-constrain (30k)", "coder7b_fs_c": "Coder-7B 3-shot",
    "ceil_zs_c": "27B zero-shot", "ceil_zs_c_retry": "27B zero-shot + retry", "ceil_fs_c": "27B 3-shot",
    "ceil_fs_c_retry": "27B 3-shot + retry", "ft2b_c": "2B FT", "ft2b_c_retry": "2B FT + retry",
    "ft2b_aug_c": "2B FT + syn", "ft2b_aug_c_retry": "2B FT + syn + retry", "ft9b_c": "9B FT",
    "ft9b_u": "9B FT, unconstrained", "ft9b_c_retry": "9B FT + retry", "ft9b_sc5": "9B FT, SC-5",
    "ft9b_sc5_retry": "9B FT, SC-5 + retry", "ft9b_aug_c": "9B FT + syn", "ft9b_aug_c_retry": "9B FT + syn + retry",
}
ORDER = list(LABEL)
MAIN = ["base9b_zs_c", "base9b_fs_c", "coder7b_fs_c", "ceil_fs_c", "ceil_zs_c_retry", "ft2b_c", "ft2b_aug_c",
        "ft9b_c", "ft9b_c_retry", "ft9b_aug_c", "ft9b_aug_c_retry"]
SHORT = {"base9b_zs_c": "9B\\,zs", "base9b_fs_c": "9B\\,3s", "coder7b_fs_c": "C7B\\,3s", "ceil_fs_c": "27B\\,3s",
         "ceil_zs_c_retry": "27B\\,zs+r", "ft2b_c": "2B\\,FT", "ft2b_aug_c": "2B\\,syn", "ft9b_c": "9B\\,FT",
         "ft9b_c_retry": "9B\\,FT+r", "ft9b_aug_c": "9B\\,syn", "ft9b_aug_c_retry": "9B\\,syn+r"}
SETNAME = {"test": "test", "test_anon": "test\\_anon", "novel_topo": "novel\\_topo", "ood": "OOD-1",
           "ood2": "OOD-2", "test_mini": "test\\_mini", "untraceable": "untraceable"}
SETORDER = ["test", "test_anon", "test_mini", "novel_topo", "ood", "ood2", "untraceable"]


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and x != x) else f"{100 * float(x):.{d}f}"


def ci(v, lo, hi):
    return f"{pct(v)}\\,{{\\tiny[{pct(lo)}, {pct(hi)}]}}"


def lab(c):
    return esc(LABEL.get(c, c))


def sort_cfg(df):
    df = df[df.config.isin(ORDER)].copy()
    df["_o"] = df.config.map(ORDER.index)
    if "eval_set" in df:
        df["_s"] = df.eval_set.map(lambda s: SETORDER.index(s) if s in SETORDER else 99)
        return df.sort_values(["_o", "_s"])
    return df.sort_values("_o")


def longtable(spec, head, rows, caption, label=None, size="\\scriptsize"):
    spec = spec.replace("p{", "L{")
    out = [f"{{{size}\\begin{{longtable}}{{{spec}}}",
           f"\\caption{{{caption}}}" + (f"\\label{{{label}}}" if label else "") + "\\\\",
           "\\toprule", head + " \\\\", "\\midrule\\endfirsthead",
           "\\toprule", head + " \\\\", "\\midrule\\endhead", "\\bottomrule\\endfoot"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out.append("\\end{longtable}}")
    return "\n".join(out)


def table(spec, head, rows, caption, label=None, size="\\scriptsize"):
    spec = spec.replace("p{", "L{")
    return "\n".join([f"\\begin{{table}}[H]\\centering{size}", f"\\begin{{tabular}}{{{spec}}}\\toprule", head + " \\\\\\midrule",
                      *[" & ".join(r) + " \\\\" for r in rows], "\\bottomrule\\end{tabular}",
                      f"\\caption{{{caption}}}" + (f"\\label{{{label}}}" if label else ""), "\\end{table}"])


def lst(text, style="plain"):
    return f"\\begin{{lstlisting}}[style={style}]\n{text}\n\\end{{lstlisting}}"


S = []          # output chunks


def sec(title, label=None):
    S.append(f"\\section{{{title}}}" + (f"\\label{{{label}}}" if label else ""))


def sub(title):
    S.append(f"\\subsection{{{title}}}")


def par(text):
    S.append(text.strip() + "\n")


# =================================================================================================================
def results():
    sec("Complete results", "app:results")
    par(r"""All numbers in this appendix are produced by \texttt{scripts/aggregate.py} (tables in \texttt{results/tables/})
or computed from the raw prediction files (\texttt{results/raw/<config>/<set>.jsonl}). Percentages are means over
samples; brackets are 95\% bootstrap confidence intervals (1{,}000 resamples). Configuration names are short labels;
Table~\ref{tab:cfgs} maps them to the configuration ids used in the code and the file names.""")

    # ---- configs
    sub("Every inference configuration")
    cf = yaml.safe_load(open(os.path.join(REPO, "configs", "experiments.yaml")))
    rows = []
    for c in cf:
        if c["name"] not in LABEL:
            continue
        model = os.path.basename(c["model_dir"]).replace("CEILING", "Qwen3.5-27B-FP8")
        samp = c.get("sampling") or {}
        dec = "greedy" if not samp else f"T={samp.get('temperature')}" + (f", top-p {samp['top_p']}" if "top_p" in samp else "")
        extras = []
        if c.get("retry_rounds"):
            extras.append(f"retry {c['retry_rounds']}")
        if c.get("self_consistency"):
            extras.append(f"SC {c['self_consistency']}")
        if c.get("thinking"):
            extras.append("thinking")
        if c.get("reasoning_parser"):
            extras.append("grammar after think")
        rows.append([lab(c["name"]), "\\texttt{" + esc(c["name"]).replace("\\_", "\\_\\allowbreak{}") + "}", esc(model),
                     "3-shot" if c["prompt_mode"] == "few_shot" else "0-shot",
                     "yes" if c.get("constrained") else "no", esc(dec), esc(", ".join(extras) or "--"),
                     str(c.get("max_tokens", 3072)), esc(", ".join(c["eval_sets"])).replace("\\_", "\\_\\allowbreak{}")])
    S.append(longtable("p{2.2cm}p{2.3cm}p{2.3cm}ccp{1.1cm}p{1.5cm}rp{1.9cm}",
                       "Label & id & model & prompt & constr. & decoding & extras & max tok. & eval sets", rows,
                       "Every configuration in \\texttt{configs/experiments.yaml} that produced reported results. "
                       "``FT'' = our merged LoRA model; ``syn'' = trained with the synthetic data; constr. = JSON-schema "
                       "constrained decoding; SC = self-consistency (n samples, the validator picks).", "tab:cfgs"))

    # ---- main test with all metrics
    sub("Official test: every headline metric with confidence intervals")
    m = sort_cfg(pd.read_csv(os.path.join(T, "main.csv")))
    rows = [[lab(r.config)] + [ci(r[k], r[k + "_lo"], r[k + "_hi"]) for k in ["fully_valid", "edge_f1", "exact_strict"]] +
            [pct(r[k]) for k in ["isomorphic", "graph_similarity", "pps", "shape_acc"]] for _, r in m.iterrows()]
    S.append(longtable("lccccccc", "Config & fully valid & edge F1 & exact & isomorphic & graph sim. & PPS & shape acc.",
                       rows, "Official test (1{,}000 samples), all headline metrics in \\% (95\\% CIs for the first three; "
                             "the CIs of all metrics are in \\texttt{results/tables/main.csv})."))

    # ---- every config x set
    sub("Every configuration on every evaluation set")
    d = sort_cfg(pd.read_csv(os.path.join(T, "by_eval_set.csv")))
    rows = [[lab(r.config), SETNAME.get(r.eval_set, esc(r.eval_set)), str(int(r.n)),
             ci(r.fully_valid, r.fully_valid_lo, r.fully_valid_hi), pct(r.edge_f1),
             ci(r.exact_strict, r.exact_strict_lo, r.exact_strict_hi), pct(r.isomorphic), pct(r.graph_similarity)]
            for _, r in d.iterrows()]
    S.append(longtable("llrccccc", "Config & set & n & fully valid & edge F1 & exact & isomorphic & graph sim.", rows,
                       "Every (configuration, evaluation set) pair that was run, in \\%. Sets not listed for a "
                       "configuration were not run."))

    # ---- per tier
    sub("By complexity tier (official test)")
    d = sort_cfg(pd.read_csv(os.path.join(T, "by_tier.csv")))
    piv_v = d.pivot(index="config", columns="tier", values="fully_valid")
    piv_e = d.pivot(index="config", columns="tier", values="exact_strict")
    piv_f = d.pivot(index="config", columns="tier", values="edge_f1")
    tiers = ["L0", "L1", "L2", "L3", "L4"]
    rows = [[lab(c)] + [pct(piv_v.loc[c, t]) for t in tiers] + [pct(piv_e.loc[c, t]) for t in tiers] +
            [pct(piv_f.loc[c, t]) for t in tiers] for c in [c for c in ORDER if c in piv_v.index]]
    S.append(longtable("l" + "c" * 15, "& \\multicolumn{5}{c}{fully valid} & \\multicolumn{5}{c}{exact} & \\multicolumn{5}{c}{edge F1}\\\\"
                       "Config & " + " & ".join(tiers * 3), rows,
                       "Official test by tier (200 samples each), in \\%. L0 = neuron MLP, L1 = chain, L2 = residual, "
                       "L3 = branch-concat, L4 = nested.", size="\\tiny"))

    # ---- by size
    sub("By graph size (official test)")
    d = sort_cfg(pd.read_csv(os.path.join(T, "by_size.csv")))
    bins = ["<=10", "11-15", "16-20", ">20"]
    pv = d.pivot(index="config", columns="size_bin", values="fully_valid")
    pe = d.pivot(index="config", columns="size_bin", values="exact_strict")
    rows = [[lab(c)] + [pct(pv.loc[c, b]) for b in bins] + [pct(pe.loc[c, b]) for b in bins]
            for c in [c for c in ORDER if c in pv.index]]
    hb = " & ".join(["$\\leq$10", "11--15", "16--20", "$>$20"] * 2)
    S.append(longtable("l" + "c" * 8, "& \\multicolumn{4}{c}{fully valid} & \\multicolumn{4}{c}{exact}\\\\ Config (nodes) & " + hb,
                       rows, "Official test by number of reference nodes, in \\%."))

    # ---- by augmentation
    sub("By augmentation variant (robustness to renaming and dead code)")
    d = sort_cfg(pd.read_csv(os.path.join(T, "by_aug.csv")))
    d = d[d.eval_set == "test"]
    augs = ["none", "aug_variant_1", "aug_variant_2", "aug_variant_3"]
    pv = d.pivot(index="config", columns="augmentation_type", values="fully_valid")
    pe = d.pivot(index="config", columns="augmentation_type", values="exact_strict")
    rows = [[lab(c)] + [pct(pv.loc[c, a]) for a in augs] + [pct(pe.loc[c, a]) for a in augs]
            for c in [c for c in ORDER if c in pv.index]]
    S.append(longtable("l" + "c" * 8, "& \\multicolumn{4}{c}{fully valid} & \\multicolumn{4}{c}{exact}\\\\ Config & orig. & renamed & dead code & all three & orig. & renamed & dead code & all three",
                       rows, "Official test by variant (250 samples each), in \\%: the original, the forward argument "
                             "renamed, dead code inserted, and all three changes plus a shuffled \\texttt{\\_\\_init\\_\\_}."))

    # ---- seen vs unseen
    sub("Seen versus held-out topologies")
    d = sort_cfg(pd.read_csv(os.path.join(T, "seen_vs_unseen.csv")))
    sl = ["test_seen", "test_unseen", "novel_topo", "ood", "ood2"]
    pv = d.pivot(index="config", columns="slice", values="fully_valid")
    pe = d.pivot(index="config", columns="slice", values="exact_strict")
    rows = [[lab(c)] + [pct(pv.loc[c, s]) if s in pv.columns else "--" for s in sl] +
            [pct(pe.loc[c, s]) if s in pe.columns else "--" for s in sl] for c in [c for c in ORDER if c in pv.index]]
    hs = " & ".join(["seen", "held-out", "novel", "OOD-1", "OOD-2"] * 2)
    S.append(longtable("l" + "c" * 10, "& \\multicolumn{5}{c}{fully valid} & \\multicolumn{5}{c}{exact}\\\\ Config & " + hs,
                       rows, "From familiar to unfamiliar, in \\%: official-test samples whose topology occurs in "
                             "train\\_ft (800), official-test samples with a held-out topology (200), novel\\_topo (564), "
                             "OOD-1 (40), OOD-2 (21). ``--'' = not run. NaN-free cells only.", size="\\tiny"))

    # ---- OOD by category
    sub("OOD-1 and OOD-2 by category")
    d = pd.read_csv(os.path.join(T, "ood_by_category.csv"))
    for es, title in (("ood", "OOD-1 (4 modules per category)"), ("ood2", "OOD-2 (3 modules per category)")):
        x = d[d.eval_set == es]
        cats = sorted(x.category.unique())
        cfgs = [c for c in MAIN if c in set(x.config)]
        pv = x.pivot(index="category", columns="config", values="fully_valid")
        pe = x.pivot(index="category", columns="config", values="exact_strict")
        rows = [[esc(cat)] + [f"{pct(pv.loc[cat, c], 0)}/{pct(pe.loc[cat, c], 0)}" for c in cfgs] for cat in cats]
        S.append(table("l" + "c" * len(cfgs), "Category & " + " & ".join(SHORT[c] for c in cfgs), rows,
                       f"{title}: fully valid / exact match, in \\%. Column key: zs = zero-shot, 3s = 3-shot, "
                       "+r = with retry, C7B = Coder-7B, syn = fine-tuned with synthetic data.", size="\\tiny"))

    # ---- validity levels
    sub("Validity levels and code-consistency rates")
    d = sort_cfg(pd.read_csv(os.path.join(T, "validity.csv")))
    cols = ["parse_ok", "schema_ok", "repaired", "struct_valid", "arity_valid", "code_consistent", "shape_consistent",
            "fully_valid", "hallucination_rate", "dead_code_inclusion_rate"]
    rows = [[lab(r.config), SETNAME.get(r.eval_set, esc(r.eval_set))] + [pct(r[k]) for k in cols] for _, r in d.iterrows()]
    S.append(longtable("ll" + "c" * len(cols), "Config & set & parse & schema & repaired & V1 & V2 & V3 & V4 & valid & halluc. & dead incl.",
                       rows, "Share of outputs passing each validator level (each level counted on its own), in \\%. "
                             "``repaired'' = the parser had to fix the JSON (e.g.\\ close a truncated bracket); "
                             "``halluc.'' = share of predicted \\texttt{call\\_module} nodes that are not declared layers; "
                             "``dead incl.'' = share of dead-code samples whose dead layer appears in the prediction.",
                       size="\\tiny"))

    # ---- retry
    sub("Verify-and-retry, every run")
    d = sort_cfg(pd.read_csv(os.path.join(T, "retry.csv")))
    rows = [[lab(r.config), SETNAME.get(r.eval_set, esc(r.eval_set)), str(int(r.n)), pct(r.initial_fully_valid),
             pct(r.final_fully_valid), f"{100 * (r.final_fully_valid - r.initial_fully_valid):+.1f}", f"{r.mean_rounds:.2f}"]
            for _, r in d.iterrows()]
    S.append(longtable("llrcccc", "Config & set & n & valid before & valid after & gain & mean rounds", rows,
                       "Verify-and-retry (at most 2 extra rounds; only invalid graphs are retried). ``Before'' is the "
                       "first attempt inside the retry run. It can differ slightly from the separate run without retry "
                       "(e.g.\\ 2B FT + syn on OOD-2: 66.7 vs 71.4), because vLLM's greedy decoding is not bit-identical "
                       "when the batch composition changes (see Section~\\ref{app:determinism})."))

    # ---- properties
    sub("Topological properties preserved (official test)")
    d = pd.read_csv(os.path.join(T, "properties.csv"))
    d = d[d.eval_set == "test"]
    props = [c for c in d.columns if c.startswith("prop_")]
    cfgs = [c for c in MAIN if c in set(d.config)]
    rows = []
    for p in props + ["pps"]:
        rows.append([esc(p.replace("prop_", "")) if p != "pps" else "\\textbf{PPS (mean)}"] +
                    [pct(d[d.config == c][p].iloc[0], 0) for c in cfgs])
    S.append(table("l" + "c" * len(cfgs), "Property & " + " & ".join(SHORT[c] for c in cfgs), rows,
                   "Share of test samples whose predicted graph has exactly the reference's value of each property, "
                   "in \\%. Definitions in Table~\\ref{tab:props}.", size="\\tiny"))

    # ---- failures
    sub("Failure labels (official test)")
    d = pd.read_csv(os.path.join(T, "failure_counts.csv"))
    d = d[d.eval_set == "test"].groupby(["config", "failures"])["count"].sum().reset_index()
    labels = ["parse_fail", "missing_skip_edge", "missing_edge", "extra_edge", "missing_node", "hallucinated_node",
              "dead_code_included", "wrong_kind", "naming_only", "shape_only", "order_error", "cycle"]
    cfgs = [c for c in MAIN if c in set(d.config)]
    rows = []
    for l_ in labels:
        rows.append([esc(l_)] + [str(int(d[(d.config == c) & (d.failures == l_)]["count"].sum())) for c in cfgs])
    S.append(table("l" + "r" * len(cfgs), "Label & " + " & ".join(SHORT[c] for c in cfgs), rows,
                   "Number of official-test samples (out of 1{,}000) with each failure label; a sample can have "
                   "several. Definitions in Table~\\ref{tab:failures}.", size="\\tiny"))

    # ---- GED
    sub("Checking the edit-distance metric against the true graph edit distance")
    g = pd.read_csv(os.path.join(T, "ged_subset.csv"))
    g["ged_exact"] = g.ged_exact.astype(str).str.lower() == "true"
    rows = []
    for c in [c for c in ORDER if c in set(g.config)]:
        x = g[g.config == c]
        rho = spearmanr(x.aed, x.ged).correlation if x.aed.nunique() > 1 and x.ged.nunique() > 1 else float("nan")
        ex = x[x.ged_exact]
        rows.append([lab(c), str(len(x)), pct(x.ged_exact.mean()), "--" if rho != rho else f"{rho:.3f}",
                     pct((ex.aed >= ex.ged).mean()) if len(ex) else "--", f"{x.aed.mean():.2f}", f"{x.ged.mean():.2f}"])
    ex = g[g.ged_exact]
    rows.append(["\\textbf{all}", str(len(g)), pct(g.ged_exact.mean()), f"{spearmanr(g.aed, g.ged).correlation:.3f}",
                 pct((ex.aed >= ex.ged).mean()), f"{g.aed.mean():.2f}", f"{g.ged.mean():.2f}"])
    S.append(longtable("lrccccc", "Config & pairs & GED exact & Spearman & AED$\\geq$GED & mean AED & mean GED", rows,
                       "True graph edit distance (networkx, 2\\,s budget per pair) vs our aligned edit distance (AED) on "
                       "a stratified 100-sample test subset per configuration (pairs that are identical are included). "
                       "``GED exact'' = the search finished within the budget (otherwise the value is an upper bound). "
                       "AED is never below an exactly computed GED, as it should be (it is the cost of one specific "
                       "edit path). ``--'' = the correlation is undefined (all distances equal)."))

    # ---- generation stats
    sub("Generation statistics: prompt and answer lengths, truncation, speed")
    gs = pd.read_csv(os.path.join(HERE, "data", "gen_stats.csv"))
    gs = sort_cfg(gs)
    rows = [[lab(r.config), SETNAME.get(r.eval_set, esc(r.eval_set)), str(int(r.n)), f"{r.prompt_tokens_mean:.0f}",
             f"{r.out_tokens_mean:.0f}", str(int(r.out_tokens_max)), pct(r.truncated), f"{r.latency_mean_s:.2f}",
             f"{r.rounds_mean:.2f}"] for _, r in gs.iterrows()]
    S.append(longtable("llrrrrccc", "Config & set & n & prompt tok. & answer tok. (mean) & max & truncated & s/sample & rounds",
                       rows, "From the raw prediction files. Answer tokens include every retry round and every "
                             "self-consistency sample. ``truncated'' = the answer hit the token limit "
                             "(\\texttt{finish\\_reason = length}). Seconds per sample are amortised over the whole "
                             "batch on one L40S (vLLM processes many prompts in parallel), so they measure throughput, "
                             "not the latency of a single request (a single 9B request takes about 9\\,s in the app)."))

    sub("Determinism and small inconsistencies")
    S.append("\\label{app:determinism}")
    par(r"""Every run used greedy decoding (temperature 0) and seed 643, except self-consistency (temperature 0.7) and
thinking mode (temperature 0.6, top-p 0.95, top-k 20, Qwen's recommended settings). Greedy decoding in vLLM is
\emph{not} bit-identical across runs when the set of prompts in a batch differs: GPU kernels add numbers in a
batch-dependent order, and a tiny difference can flip one token, after which the answer diverges. This explains the
few places where two runs that should agree differ by one or two samples: FT-2B + synthetic on OOD-2 is 71.4\% in its
own run but 66.7\% as the first attempt of the retry run; FT-9B is 99.7\% on test, while the first attempt inside the
FT-9B retry run is 99.6\%; FT-2B + retry ends at 93.2\% on test, below the 93.3\% of the run without retry. These
differences are one or two samples and well inside the confidence intervals. The 9B 3-shot unconstrained run (54.6\%)
vs constrained (54.9\%) is a difference of the same size.""")


# =================================================================================================================
def data_details():
    sec("Data details", "app:data")
    sub("The given dataset, per tier")
    md = open(os.path.join(T, "dataset_stats.md")).read()
    rows = []
    for line in md.splitlines():
        mm = re.match(r"\| (L\d) \| (\d+) \| (\d+) \| (\d+) \| ([\d/]+) \| ([\d/]+) \| (\d+) \| (\d+) \|", line)
        if mm:
            rows.append(list(mm.groups()))
    S.append(table("lrrrccrr", "Tier & samples & train & test & nodes min/med/max & edges min/med/max & structures & kind-labelled",
                   rows, "Per-tier statistics of the 5{,}000 samples. ``Structures'' ignore names and kinds (54 in "
                         "total); ``kind-labelled'' structures also distinguish node kinds (286 in total; L1 varies most "
                         "because its activations and normalisations vary). Every one of the 53 structures in the "
                         "official test also occurs in the official train split.", size="\\footnotesize"))

    sub("Splits and the novel-topology holdout")
    sizes = {}
    for line in md.split("## Splits")[1].split("##")[0].splitlines():
        mm = re.match(r"\| (\w+) \| (\d+) \|", line)
        if mm and mm.group(1) != "split":
            sizes[mm.group(1)] = mm.group(2)
    tiers = defaultdict(dict)
    for es in ["test", "test_anon", "test_mini", "novel_topo", "val"]:
        p = os.path.join(PROC, f"{es}.jsonl")
        if os.path.exists(p):
            for r in map(json.loads, open(p)):
                tiers[es][r["tier"]] = tiers[es].get(r["tier"], 0) + 1
    rows = []
    desc = {"train_ft": "fine-tuning data: official train minus held-out structures minus val",
            "val": "validation (5\\% of families; loss curve and sanity check only)",
            "novel_topo": "official-train samples whose structure was held out of train\\_ft",
            "test": "official test, unchanged", "test_anon": "official test, every class renamed to \\texttt{Net}",
            "test_mini": "stratified 100-sample subset of test (expensive experiments)",
            "test_struct_unseen": "the official-test samples whose structure was held out"}
    for k, v in sizes.items():
        t = tiers.get(k, {})
        rows.append([esc(k), v, " / ".join(str(t.get(x, "--")) for x in ["L0", "L1", "L2", "L3", "L4"]) if t else "--",
                     desc.get(k, "")])
    S.append(table("lrcp{6.6cm}", "Split & samples & per tier (L0/L1/L2/L3/L4) & purpose", rows,
                   "Splits (seed 643; all ids are stored in \\texttt{data/processed/splits.json}).", size="\\footnotesize"))
    hold = []
    for line in md.split("## Novel-topology holdout")[1].split("##")[0].splitlines():
        mm = re.match(r"\| (L\d) \| (\d+) \| (\d+) \| \[(.*)\] \|", line)
        if mm:
            hold.append([mm.group(1), mm.group(2), mm.group(3), esc(mm.group(4))])
    S.append(table("lccl", "Tier & structure classes in train & held out & samples per held-out class", hold,
                   "The novel-topology holdout: 11 structure classes chosen with a fixed seed (never the largest class "
                   "of a tier). All samples with these structures were removed from train\\_ft.", size="\\footnotesize"))

    sub("Node kinds, token budget, few-shot examples")
    voc = md.split("## Kind vocabulary")[1].split("\n\n")[1].strip()
    par("\\textbf{Most frequent node kinds} (node counts over all 5{,}000 graphs): " +
        esc(voc.replace("`", "")) + ". The full vocabulary is small: the dataset uses a handful of layer types.")
    ta = json.load(open(os.path.join(PROC, "token_audit.json")))
    tr, va, fs = ta["train"], ta["val"], ta["few_shot_prompt"]
    par("\\textbf{Token audit} (Qwen3.5 tokenizer). Training examples (prompt + answer): mean %.0f, 99th percentile %d, "
        "maximum %d tokens over %s samples; validation maximum %d. The training \\texttt{max\\_length} is %s, so no "
        "example was truncated. Three-shot prompts are up to %s tokens (mean %.0f). The answer ends with the turn-end "
        "token \\texttt{<|im\\_end|>} (id 248046); the 0.8B, 2B and 9B chat templates are byte-identical, so one set of "
        "training files serves all sizes."
        % (tr["mean"], tr["p99"], tr["max"], f"{tr['n']:,}".replace(",", "{,}"), va["max"],
           f"{ta['max_length']:,}".replace(",", "{,}"), f"{fs['max']:,}".replace(",", "{,}"), fs["mean"]))
    sp = json.load(open(os.path.join(PROC, "splits.json")))
    par("\\textbf{Few-shot examples} (always the same three, from train\\_ft, original variant): " +
        ", ".join(f"\\texttt{{{esc(x)}}}" for x in sp["few_shot_exemplars"]) +
        " (one L0, one L2, one L4). They are never evaluation samples.")

    sub("OOD-1, OOD-2 and the untraceable modules")
    for es, title in (("ood", "OOD-1"), ("ood2", "OOD-2")):
        rows = []
        for r in map(json.loads, open(os.path.join(PROC, f"{es}.jsonl"))):
            shp = r["input_shape"]
            shp = " + ".join(str(s) for s in shp) if shp and isinstance(shp[0], list) else str(shp)
            rows.append([f"\\texttt{{{esc(r['sample_id'])}}}", esc(r["category"]), str(len(r["target_graph"]["nodes"])),
                         esc(shp), str(len(r["source_code"].splitlines()))])
        S.append(longtable("llrll", "Module & category & nodes & input shape & code lines", rows,
                           f"{title} modules (source code in \\texttt{{data/{'ood' if es == 'ood' else 'ood2'}/}}). "
                           "The validator reports zero errors on every gold graph.", size="\\scriptsize"))
    unt = {"UNT_assert_value": "an \\texttt{assert} on a tensor value", "UNT_data_dependent_if": "an \\texttt{if} on a tensor value",
           "UNT_data_dependent_loop_count": "a loop count computed from a tensor", "UNT_early_exit": "a \\texttt{return} inside a data-dependent \\texttt{if}",
           "UNT_int_shape": "\\texttt{int(...)} of a traced shape value", "UNT_while_loop": "a \\texttt{while} loop on a tensor condition"}
    rows = [[f"\\texttt{{{esc(r['sample_id'])}}}", unt.get(r["sample_id"], ""), esc((r.get("oracle_error") or "").split(":")[0])]
            for r in map(json.loads, open(os.path.join(PROC, "untraceable.jsonl")))]
    S.append(table("llll"[:3], "Module & construct that breaks tracing & torch.fx error", rows,
                   "The six untraceable modules. There is no reference graph; they are used only to show what the models "
                   "do (Example 12).", size="\\footnotesize"))

    sub("The synthetic data generator")
    par(r"""\texttt{scripts/gen\_synthetic.py} builds random modules from these building blocks, each with random channel
counts (4--32), random kernel sizes and random attribute names:
\begin{itemize}[leftmargin=*]
\item \textbf{units}: convolution (optionally strided) with optional batch norm and one of several activations
(module or functional form); dropout; pooling (max/avg);
\item \textbf{blocks}: residual (identity or 1$\times$1 projection skip), branch-and-concatenate (2--3 branches),
U-Net skip (down-sample, process, up-sample, concatenate), dense block (repeated concatenation), squeeze-and-excitation
gating (\texttt{mean}, \texttt{Linear}, \texttt{sigmoid}, broadcasting multiply), \texttt{nn.Sequential} stacks,
\texttt{nn.ModuleList} loops, nested block classes (attribute paths like \texttt{stage1.conv});
\item \textbf{heads and bodies}: convolutional head (pool, flatten, \texttt{Linear}), MLP bodies with LayerNorm, GELU and
Dropout, two-input fusion models;
\item \textbf{noise}: random class names (\texttt{Net}, \texttt{Encoder}, \texttt{Tower}, \dots), random forward-argument
names (\texttt{x}, \texttt{inp}, \texttt{feats}, \dots), dead layers, shuffled \texttt{\_\_init\_\_} order.
\end{itemize}
A module is kept only if torch.fx traces it, the validator reports zero errors on its gold graph, it has at most 40
nodes and fits the token budget, and its structure is not one of the novel\_topo or OOD-1 structures (also not an
OOD-1 graph's kind-labelled structure). Numbers: 3{,}200 generated with seed 6430 (264 candidates rejected for a
protected structure on the way), 2{,}090 distinct structures, 2{,}809 within the token budget and added to training
(5{,}877 training samples in total). The leak audit (\texttt{scripts/audit\_synthetic.py}) found zero novel\_topo or
OOD-1 graphs sharing a structure with any trained synthetic module; the closest source-code pair has a token Jaccard
similarity of 0.66 (median of the closest pairs 0.53) and is visibly different code. OOD-2 was written after the
generator was frozen.""")


# =================================================================================================================
def prompts_section():
    sec("Prompts and output format", "app:prompts")
    sub("The system prompt (identical for every model, training and inference)")
    S.append(lst(P.SYSTEM_PROMPT))
    sub("User message, few-shot layout, retry message")
    par(r"""\textbf{User message.} The source code inside a Python code block; for the multi-input OOD modules (whose
input order rule 7 cannot fix) one extra line gives the input shapes:""")
    S.append(lst(P.USER_TEMPLATE.replace("{code}", "<source code>") + "\nInput shape: <only for multi-input modules>"))
    par(r"""\textbf{Few-shot layout.} System prompt, then three (user, assistant) pairs with the three fixed training
examples and their exact CG-JSON answers, then the real user message. \textbf{Retry message} (sent as a new user turn
after the model's previous answer; at most 12 error messages, at most 2 rounds):""")
    S.append(lst(P.RETRY_TEMPLATE.replace("{errors}", "- <validator message 1>\n- <validator message 2>")))
    par(r"""\textbf{Chat rendering.} Messages are rendered with each model's own chat template. For Qwen3.5 with
thinking off, the template ends the prompt with \texttt{<|im\_start|>assistant}, a newline and an empty
\texttt{<think></think>} block, so the model starts directly with the JSON. With thinking on, the model writes its
reasoning first; an answer counts only if the reasoning was closed with \texttt{</think>}. The training answers are
the compact one-line CG-JSON followed by exactly one \texttt{<|im\_end|>} token (checked on the trainer's own batch).""")
    sub("The JSON schema used for constrained decoding")
    S.append(lst(open(os.path.join(REPO, "configs", "graph_schema.json")).read().strip()))
    par(r"""\textbf{Parser} (\texttt{tgraph/parse.py}), applied to every raw answer: remove a \texttt{<think>} block (for
thinking runs, no \texttt{</think>} means no answer), strip code fences, take the first balanced \texttt{\{\dots\}},
parse it, and if that fails repair it (close brackets of a truncated answer) with \texttt{json\_repair}; numbers written as
strings (\texttt{"16"}) are converted; then the schema is checked. Constrained runs always parse (the grammar guarantees it), so their 100\% parse rate is by construction and
not a result.""")


# =================================================================================================================
def validator_metrics():
    sec("Validator and metric reference", "app:validator")
    sub("Every validator check")
    rows = [
        ("V0", "PARSE", "no JSON object could be extracted"), ("V0", "SCHEMA", "JSON does not match the schema (e.g.\\ a node without \\texttt{shape})"),
        ("V1", "DUP\\_NAME", "node \\dots{} duplicates the name of node \\dots"), ("V1", "BAD\\_EDGE", "edge [a, b] points outside the n nodes"),
        ("V1", "SELF\\_LOOP", "edge [a, a] is a self-loop"), ("V1", "DUP\\_EDGE", "edge [a, b] is listed twice"),
        ("V1", "NO\\_INPUT", "graph has no placeholder (input) node"), ("V1", "OUTPUT\\_COUNT", "graph needs exactly one output node as the last node"),
        ("V1", "INPUT\\_HAS\\_PRED", "placeholder has incoming edges"), ("V1", "OUTPUT\\_DEGREE", "output node needs $\\geq$1 input and no outgoing edges"),
        ("V1", "CYCLE", "graph has a cycle through nodes \\dots"), ("V1", "ORDER", "edge goes backwards: nodes must be listed in execution order"),
        ("V1", "ORPHAN", "node is not reachable from any input"), ("V1", "DEAD\\_END", "node does not lead to the output (dead end or dead code)"),
        ("V1", "OP\\_KIND", "op and kind disagree (e.g.\\ op \\texttt{placeholder} with kind \\texttt{input})"),
        ("V2", "ARITY", "wrong number of inputs: binary functions such as \\texttt{add}/\\texttt{mul} need 2 tensors (one input is only a warning if the code contains a tensor--scalar operation); joining functions such as \\texttt{cat} need $\\geq$2; single-input functions and layers exactly 1; the output node exactly 1 (unless \\texttt{forward} returns a tuple)"),
        ("V3", "UNDECLARED\\_MODULE", "a \\texttt{call\\_module} node that is not a layer declared in \\texttt{\\_\\_init\\_\\_} (hallucinated)"),
        ("V3", "KIND\\_MISMATCH", "node has kind X but \\texttt{self.<attr>} is a Y"),
        ("V3", "MISSING\\_MODULE", "a layer is called k times in \\texttt{forward} but appears fewer times in the graph"),
        ("V3", "DEAD\\_CODE\\_INCLUDED", "a layer whose result never reaches the \\texttt{return} appears in the graph"),
        ("V4", "SHAPE", "input shape differs from the convention; conv/linear output channels differ from \\texttt{out\\_channels}/\\texttt{out\\_features}; shape-preserving ops change the shape; reshapes change the element count; \\texttt{add}/\\texttt{mul} inputs not broadcast-compatible or output not their broadcast; \\texttt{cat} output not the concatenation"),
    ]
    rows = [(a, b.replace("\\_", "\\_\\allowbreak{}"), c) for a, b, c in rows]
    S.append(longtable("lp{3.9cm}p{11.0cm}", "Level & code & what it means / message", [list(r) for r in rows],
                       "All checks in \\texttt{tgraph/validate.py}. They use only the predicted graph and the source "
                       "code (read with Python's \\texttt{ast}, never executed), never the reference. A level passes when "
                       "none of its errors occur; \\emph{fully valid} = V0--V4 all pass. If the code analysis itself fails, "
                       "V3 is skipped with a warning.", "tab:checks", size="\\footnotesize"))

    sub("Metric definitions")
    rows = [
        ("node F1", "nodes matched by exact name; a matched node is correct if its op and kind also agree; F1 of precision and recall"),
        ("node F1 (aligned)", "same, after also pairing unmatched nodes of the same kind by the Hungarian algorithm (cost = difference in graph depth)"),
        ("edge F1", "edges as (source name, target name) pairs; F1 against the reference edge set"),
        ("edge F1 (aligned)", "edges mapped through the aligned node matching"),
        ("exact match", "same multiset of (name, op, kind) and the same named edge set; shapes are ignored"),
        ("isomorphic", "same graph up to renaming, with node kinds as labels (networkx VF2)"),
        ("WL-equal", "equal Weisfeiler--Lehman hashes (3 iterations, kinds as labels): a fast approximate isomorphism test"),
        ("shape accuracy", "share of name-matched nodes whose shape equals the reference shape"),
        ("AED", "aligned edit distance: unmatched nodes on both sides + matched nodes with a different kind + edges in exactly one of the two graphs (after mapping) + predicted edges touching unmatched nodes"),
        ("graph similarity", "$1 - \\min(1, \\text{AED} / (|V_p|+|V_r|+|E_p|+|E_r|))$"),
        ("PPS", "property preservation score: share of the 14 properties below whose value equals the reference's"),
        ("reach F1", "F1 of the transitive reachability pairs ($u \\rightsquigarrow v$) under the aligned matching: rewards connectivity even when one edge is misplaced"),
        ("kind / typed-edge multiset F1", "F1 of the multisets of node kinds and of (kind, kind) edge pairs: name-free"),
        ("fully valid", "passes validator levels V0--V4 (no reference needed)"),
    ]
    S.append(longtable("p{3.4cm}p{12.6cm}"[:0] + "p{3.3cm}p{12.6cm}", "Metric & definition", [list(r) for r in rows],
                       "Per-sample metrics (\\texttt{tgraph/metrics.py}); a parse failure scores 0 on everything. Before "
                       "comparing, a single input node is renamed to \\texttt{x} in both graphs (the dataset convention), so "
                       "the forward argument's name does not matter.", size="\\footnotesize"))
    rows = [("n\\_nodes, n\\_edges", "number of nodes / edges"), ("n\\_sources, n\\_sinks", "nodes without inputs / outputs"),
            ("n\\_branch, n\\_merge", "nodes with out-degree $>$1 / in-degree $>$1"),
            ("depth", "longest path from an input (in levels)"), ("width", "largest topological generation"),
            ("n\\_skip\\_edges", "edges that jump over at least one level (residual and dense connections)"),
            ("n\\_paths\\_in\\_out", "number of input-to-output paths (capped at 10{,}000)"),
            ("n\\_cycles", "number of directed cycles (0 for a valid graph; capped at 100)"),
            ("hierarchy\\_groups", "name prefixes shared by at least two layer nodes (\\texttt{fc1\\_0}, \\texttt{fc1\\_1}, \\dots; or dotted scopes)"),
            ("is\\_dag, weakly\\_connected", "acyclic / connected ignoring direction")]
    S.append(table("p{4.2cm}p{11.6cm}", "Property & definition", [list(r) for r in rows],
                   "The 14 topological properties (\\texttt{tgraph/properties.py}).", "tab:props", size="\\footnotesize"))
    rows = [("parse\\_fail", "no graph could be extracted"),
            ("missing\\_skip\\_edge", "a reference edge that jumps $>$1 level is absent"),
            ("missing\\_edge", "a reference edge between adjacent levels is absent"),
            ("extra\\_edge", "a predicted edge that is not in the reference (or touches an unmatched node)"),
            ("missing\\_node", "a reference node has no counterpart"), ("hallucinated\\_node", "a predicted node has no counterpart"),
            ("dead\\_code\\_included", "the validator found a dead layer in the prediction"),
            ("wrong\\_kind", "a matched node has the wrong kind (e.g.\\ \\texttt{ReLU} vs \\texttt{relu})"),
            ("naming\\_only", "the graph is isomorphic to the reference but some names differ"),
            ("shape\\_only", "exact topology, but some shapes are wrong"),
            ("order\\_error", "edges go backwards in the node list"), ("cycle", "the graph has a cycle")]
    S.append(table("p{3.6cm}p{12.2cm}", "Label & rule", [list(r) for r in rows],
                   "Failure labels (\\texttt{tgraph/failures.py}); a sample can get several.", "tab:failures",
                   size="\\footnotesize"))


# =================================================================================================================
def training_inference():
    sec("Training and inference settings", "app:training")
    rows = [("method", "supervised fine-tuning with LoRA (bf16, not quantised), TRL \\texttt{SFTTrainer}; loss only on the answer tokens"),
            ("LoRA", "rank 32, alpha 64, dropout 0.05"),
            ("LoRA targets", "\\texttt{q\\_proj k\\_proj v\\_proj o\\_proj gate\\_proj up\\_proj down\\_proj} (attention and MLP) and \\texttt{in\\_proj\\_qkv in\\_proj\\_z out\\_proj} (Gated DeltaNet linear-attention layers); not \\texttt{in\\_proj\\_a/b} (one value per head), not the vision tower, not the multi-token-prediction head"),
            ("trainable parameters", "2B: 31.3\\,M; 9B: 80.2\\,M (checked: 0 in the vision tower)"),
            ("schedule", "2 epochs, cosine learning-rate decay, 3\\% warm-up; learning rate $2\\cdot10^{-4}$ (2B) and $10^{-4}$ (9B)"),
            ("batch", "effective 16: 2B 8 per step $\\times$ 2 accumulation; 9B 2 $\\times$ 8"),
            ("sequence", "\\texttt{max\\_length} 2{,}240 tokens, no packing, no truncation; gradient checkpointing (non-reentrant)"),
            ("checkpoints", "evaluation and save every 50 steps, last 2 kept; seed 643"),
            ("data", "train\\_ft (3{,}068) or train\\_ft + 2{,}809 synthetic (5{,}877); validation 160"),
            ("merge", "\\texttt{scripts/merge\\_lora.py} adds $\\frac{\\alpha}{r} BA$ to each adapted weight directly in the safetensors files, so the merged folder has the base model's exact layout (vLLM loads it like the base model). Checked on 0.8B with random adapters: merged and adapter logits agree within 0.1 (vs an adapter effect of 11.7)"),
            ("sanity check", "after training, 20 validation samples are decoded greedily: 20/20 exact for all four adapters")]
    S.append(table("p{3.2cm}p{12.6cm}", "Setting & value", [list(r) for r in rows],
                   "Fine-tuning settings (\\texttt{scripts/train\\_sft.py}, \\texttt{configs/train\\_*.yaml}).",
                   size="\\footnotesize"))
    stats = []
    for tag, name in (("9b_aug", "9B + synthetic"), ("9b", "9B"), ("2b_aug", "2B + synthetic"), ("2b", "2B")):
        rows_ = list(csv.DictReader(open(os.path.join(REPO, "runs", f"train_{tag}", "log_history.csv"))))
        tr = [r for r in rows_ if r.get("loss")]
        ev = [r for r in rows_ if r.get("eval_loss")]
        last = rows_[-1]
        steps = int(max(float(r["step"]) for r in rows_ if r.get("step")))
        stats.append([name, str(steps), f"{float(last['train_runtime']) / 3600:.2f}\\,h",
                      f"{float(last['train_runtime']) / steps:.1f}\\,s", f"{float(tr[0]['loss']):.3f}",
                      f"{float(tr[-1]['loss']):.1e}", f"{float(ev[-1]['eval_loss']):.1e}"])
    S.append(table("lrrrrrr", "Adapter & steps & time & s/step & first loss & last loss & val loss", stats,
                   "Training runs (one NVIDIA L40S each), from \\texttt{runs/train\\_*/log\\_history.csv}. Losses near zero "
                   "are expected: the answer is fully determined by the code.", size="\\footnotesize"))
    rows = [("engine", "vLLM 0.26.0, bf16 weights (27B: the official FP8 checkpoint), one GPU"),
            ("context", "\\texttt{max\\_model\\_len} 8{,}192 (thinking: 12{,}288 or 34{,}816)"),
            ("answer budget", "3{,}072 tokens (thinking: 8{,}192 or 30{,}000)"),
            ("decoding", "greedy, seed 643; self-consistency T = 0.7; thinking T = 0.6, top-p 0.95, top-k 20"),
            ("constrained decoding", "vLLM structured outputs with the JSON schema above; retry rounds are always constrained; for think-then-constrain the grammar applies only after \\texttt{</think>} (\\texttt{reasoning\\_parser=qwen3})"),
            ("memory", "\\texttt{gpu\\_memory\\_utilization} 0.88 (27B: 0.92 and at most 64 parallel sequences, because Qwen3.5's recurrent-state cache left room for only 151 sequences next to the 27B weights)"),
            ("other", "prefix caching off (vLLM marks it experimental for Qwen3.5's linear-attention layers); multimodal inputs disabled; vLLM's PyTorch sampler (the default flashinfer sampler needs a CUDA compiler the cluster lacks)"),
            ("batching", "all prompts of an evaluation set in one call; retry rounds re-submit only the invalid answers"),
            ("HF backend", "used by the app's CPU mode: transformers \\texttt{generate}, greedy, no constrained decoding")]
    S.append(table("p{3.2cm}p{12.6cm}", "Setting & value", [list(r) for r in rows],
                   "Inference settings (\\texttt{scripts/generate.py}).", size="\\footnotesize"))


# =================================================================================================================
def environment():
    sec("Software, hardware and compute", "app:env")
    rows = [("Python", "3.11", "3.11"), ("PyTorch", "2.11.0+cu128", "2.11.0+cu129"), ("transformers", "5.19.0", "5.20.2"),
            ("TRL / PEFT / accelerate", "1.16.0 / 0.21.2 / 1.15.0", "-- / 0.21.2 / 1.15.0"),
            ("flash-linear-attention", "0.5.2", "--"), ("vLLM", "--", "0.26.0+cu129"), ("xgrammar / flashinfer", "--", "0.2.7 / 0.6.14"),
            ("torchcodec", "--", "0.15.0+cu129 (+ conda ffmpeg $<$8)"),
            ("networkx / scipy / numpy", "3.6.1 / 1.17.1 / 2.4.6", "3.6.1 / 1.17.1 / 2.3.5"),
            ("gradio / matplotlib / graphviz", "6.30.0 / 3.11.2 / 0.21", "6.30.0 / 3.11.2 / 0.21"),
            ("json\\_repair / jsonschema", "0.64.2 / 4.26.0", "0.64.2 / 4.26.0")]
    S.append(table("lll", "Package & tg-train & tg-infer", [list(r) for r in rows],
                   "The two conda environments (exact pins: \\texttt{requirements-*.lock.txt}). Two are needed because "
                   "vLLM pins its own PyTorch build.", size="\\footnotesize"))
    par(r"""\textbf{Hardware and cluster.} IIT Bombay's Prajna cluster: NVIDIA L40S GPUs (48\,GB, 44.4\,GiB usable), driver
570.86.15 (CUDA $\leq$ 12.8), Lustre file system, no internet on compute nodes (every job runs with the Hugging Face
offline flags). SLURM rules that shaped the work: \texttt{--qos} must equal the partition; the \texttt{l40} partition
allows 4 running and 5 submitted jobs per user (two inference jobs were merged for that reason); only the
\texttt{interactive} partition accepts \texttt{srun} (used for CPU work and the live demo; node \texttt{cn22-a40} was
excluded after "Job credential expired" errors); compute nodes cannot be reached by \texttt{ssh} (the app is reached by
port forwarding through the login node).

\textbf{Isolation.} The same account runs another research project whose evaluation depends on bit-exact caches. All
of this project's files live under \texttt{\textasciitilde/IE643\_Project}: \texttt{tgenv.sh} redirects the conda
package cache, \texttt{PIP\_CACHE\_DIR}, \texttt{HF\_HOME}, \texttt{TRITON\_CACHE\_DIR}, \texttt{TMPDIR},
\texttt{XDG\_CACHE\_HOME}, \texttt{VLLM\_CACHE\_ROOT}, \texttt{FLASHINFER\_WORKSPACE\_BASE},
\texttt{TORCHINDUCTOR\_CACHE\_DIR} and \texttt{TORCH\_HOME} there. The base Qwen3.5-9B weights are shared read-only
through a symbolic link; all other models were downloaded into \texttt{models/}.""")
    # compute log
    rows, gpu_s, cpu_s = [], 0, 0
    for r in csv.DictReader(open(os.path.join(HERE, "data", "compute_log.psv")), delimiter="|"):
        h, mi, s = (int(x) for x in r["Elapsed"].split("-")[-1].split(":"))
        sec_ = h * 3600 + mi * 60 + s
        gpu = "gpu" in r["AllocTRES"]
        if gpu:
            gpu_s += sec_
        else:
            cpu_s += sec_
        state = r["State"].replace("CANCELLED by 10857", "CANCELLED")
        rows.append([r["JobID"], esc(r["JobName"]), r["Partition"], "1 GPU" if gpu else "CPU", esc(state), r["Elapsed"],
                     esc(r["Start"].replace("T", " ")[5:16]), esc(r["NodeList"])])
    S.append(longtable("rlllllll", "Job & name & partition & resources & state & elapsed & start & node", rows,
                       f"Every SLURM job of the project (\\texttt{{sacct}}). Total GPU time {gpu_s / 3600:.1f}\\,GPU-hours "
                       f"(all on single L40S GPUs; this includes failed, cancelled and demo jobs), CPU-only jobs "
                       f"{cpu_s / 3600:.1f}\\,h on 16 cores. Cancelled jobs are the restart after the prompt fix "
                       "(Section~\\ref{sec:journey}), a ceiling attempt that needed a memory setting, and test runs. "
                       "\\texttt{tg-app} is the live-demo test (ended by its 1-hour limit).", "tab:compute"))


# =================================================================================================================
def decisions_issues_timeline():
    sec("Decision log", "app:decisions")
    rows = [
        ("1", "Environments and caches", "two new conda envs and every cache inside the project folder; the 9B weights reused read-only", "do not disturb the other project on the same account"),
        ("2", "vLLM 0.26.0 + CUDA 12.9 build", "newest vLLM with a PyTorch build that runs on the CUDA-12.8 driver", "vLLM $\\geq$0.27 needs PyTorch 2.13 (no CUDA-12.8 build)"),
        ("3", "training env: PyTorch 2.11 cu128 + fla 0.5.2", "combination already known to run Qwen3.5 on this cluster; \\texttt{causal-conv1d} not installed (needs a CUDA compiler)", "lowest risk"),
        ("4", "models as plain folders", "downloaded with curl into \\texttt{models/<name>}, loaded by path", "compute nodes are offline"),
        ("5", "54 structures, not $\\sim$68", "the spec double-counted structures shared between tiers", "measured with networkx"),
        ("6", "novel-topology holdout", "1/4/3/2/1 classes per tier held out (seed 643); 564 samples", "test generalisation to unseen topologies"),
        ("7", "\\texttt{max\\_length} 2{,}240", "the longest training example is 2{,}157 tokens", "no truncation"),
        ("8", "canonical prompt", "ends with an empty think block (template behaviour with thinking off)", "byte-identical train/inference prompts"),
        ("9", "oracle sandbox", "one long-lived spawned worker with a 20\\,s per-call limit", "importing torch takes $>$20\\,s on Lustre; \\texttt{fork} deadlocks"),
        ("10", "OOD-1 set", "40 modules in 10 categories; \\texttt{se\\_3} got a conv stem", "rule 7 was ambiguous without it"),
        ("11", "validator: scalar arithmetic", "\\texttt{add}/\\texttt{mul} with one tensor input is an error unless the code has a tensor--scalar operation (then a warning)", "zero false alarms on 5{,}000 gold graphs"),
        ("11b", "prefix caching off", "vLLM marks it experimental for Qwen3.5's linear-attention layers", "results must not depend on it"),
        ("12", "failure label \\texttt{missing\\_edge}", "added next to \\texttt{missing\\_skip\\_edge}", "a dropped chain edge had no label (found in the dry run)"),
        ("13", "prompt identity across envs", "checked 200/200 identical prompts between the two transformers versions", "training and inference use different envs"),
        ("14", "prompt v2", "rule 1 states the input node's kind \\texttt{placeholder}; compact one-line JSON; 3{,}072-token budget; everything re-run", "957/1{,}000 zero-shot answers failed only on that kind; 12\\% ran out of tokens"),
        ("15", "thinking mode", "an answer needs \\texttt{</think>}; added a 30k-token run", "97\\% unfinished at 8k; the parser had taken fragments from unfinished reasoning"),
        ("16", "oracle warm-up", "the worker traces a tiny model before reporting ready; untraceable set skipped in bulk evaluation", "first-use imports counted against the 20\\,s limit"),
        ("17", "night plan", "think-then-constrain, self-consistency, the synthetic-data arm", "the remaining weakness was generalisation"),
        ("18", "ceiling = Qwen3.5-27B-FP8", "official FP8 checkpoint (29\\,GB) fits one L40S; bf16 (55.6\\,GB) does not", "same family as the main model"),
        ("19", "OOD-2", "21 modules written after the generator was frozen, with operators absent from dataset and generator; 3 tuple-output modules dropped", "an honest generalisation test for the synthetic-data models"),
    ]
    S.append(longtable("lp{3.6cm}p{6.6cm}p{5.0cm}", "\\# & decision & what & why", [list(r) for r in rows],
                       "Every deviation from (or choice within) the original build spec, condensed from "
                       "\\texttt{DECISIONS.md}.", size="\\scriptsize"))

    sec("Problems encountered and how they were fixed", "app:issues")
    rows = [
        ("two conda builds at once", "both failed (\\texttt{InvalidArchiveError}): they share one package cache", "rebuilt one after the other"),
        ("\\texttt{import vllm} (1)", "torchcodec from PyPI is built for CUDA 13", "torchcodec 0.15.0+cu129"),
        ("\\texttt{import vllm} (2)", "torchcodec needs FFmpeg shared libraries", "conda-forge \\texttt{ffmpeg<8}"),
        ("\\texttt{import vllm} (3)", "FFmpeg's ICU needs a newer C++ runtime than the system's", "\\texttt{LD\\_PRELOAD} of the env's \\texttt{libstdc++}"),
        ("vLLM engine start", "flashinfer's sampler compiles CUDA code at run time; no compiler on the cluster", "\\texttt{VLLM\\_USE\\_FLASHINFER\\_SAMPLER=0}"),
        ("cache leak", "vLLM and flashinfer wrote into \\texttt{\\textasciitilde/.cache}", "moved into the project; extra cache variables in \\texttt{tgenv.sh}"),
        ("Triton on CPU", "transformers picks the fla GPU kernels even without a GPU", "\\texttt{cpu\\_disable\\_fla()} on CPU-only runs"),
        ("CPU training", "one step of 870 tokens did not finish in 25 min", "all training on GPUs"),
        ("TRL/transformers API changes", "\\texttt{warmup\\_ratio} and \\texttt{logging\\_dir} removed; duplicate end token", "\\texttt{warmup\\_steps=0.03}, env variable for logs, \\texttt{eos\\_token=<|im\\_end|>}"),
        ("node cn22-a40", "\"Job credential expired\" on every launch", "excluded"),
        ("oracle subprocess", "a fresh process per call exceeded 20\\,s; \\texttt{fork} deadlocked", "one persistent spawned worker, with warm-up"),
        ("dry run", "a dropped ordinary edge got no failure label", "added \\texttt{missing\\_edge}"),
        ("prompt v1", "0\\% zero-shot validity (input-node kind; pretty-printed JSON ran out of tokens)", "prompt v2 with the user's OK; all jobs restarted"),
        ("job limits", "\\texttt{l40} allows only 5 submitted jobs per user", "two inference jobs merged, with a dependency on both trainings"),
        ("thinking answers", "8k tokens too few; fragments parsed from unfinished reasoning", "\\texttt{</think>} required, \\texttt{scripts/reparse.py}, 30k-token run"),
        ("27B ceiling", "default 256 parallel sequences exceeded the recurrent-state cache", "\\texttt{max\\_num\\_seqs} 64, memory 0.92"),
        ("OOD \\texttt{se\\_3}", "the input-shape rule was ambiguous", "added a conv stem"),
        ("tuple outputs", "\\texttt{chunk}/\\texttt{split} results have no tensor shape in the schema", "3 OOD-2 modules dropped"),
        ("untraceable metrics", "no reference, so no metrics", "skipped in bulk evaluation; shown qualitatively"),
        ("app first click", "about 90\\,s (oracle worker start + grammar compilation)", "\\texttt{--preload} sends one warm-up request"),
    ]
    S.append(longtable("p{3.2cm}p{7.0cm}p{5.6cm}", "Problem & cause & fix", [list(r) for r in rows],
                       "Problems met during the project, in order (details in \\texttt{PROJECT\\_LOG.md}).", size="\\scriptsize"))

    sec("Timeline", "app:timeline")
    rows = [
        ("24 Oct, afternoon", "plan read; environments built; models downloaded; core package, OOD-1 and tests written"),
        ("24 Oct, evening", "all CPU tests green (oracle 5{,}000/5{,}000, validator silent on gold); vLLM import repairs; CPU smoke"),
        ("24 Oct, 20:05--22:30", "GPU smoke (two attempts): vLLM, training speed (9B 12.1\\,s/step), merge check, evaluation dry run"),
        ("24 Oct, 23:21", "full runs submitted on the user's ``submit'' (baselines, 2B and 9B training)"),
        ("24 Oct, 23:40", "prompt fix (v2) after the first zero-shot results; all affected jobs restarted"),
        ("25 Oct, 00:45", "first results: 9B 3-shot 54.9\\%, FT-2B 93.3\\% (0\\% on the held-out L4 topology), thinking needs more tokens"),
        ("25 Oct, 02:00--03:30", "FT-9B 99.7\\% on test but 47.5\\% on OOD-1; night plan: synthetic data, 27B ceiling, retry, self-consistency, think-then-constrain"),
        ("25 Oct, 03:50", "OOD-2 written after the generator was frozen"),
        ("25 Oct, 06:07", "last GPU job finished"),
        ("25 Oct, 06:40", "final evaluation on CPU: metrics, GED check, tables, figures, galleries, app cache"),
        ("25 Oct, daytime", "explainer, live GPU demo test, README, model card, CPU-mode and adapter checks, submission zip"),
    ]
    S.append(table("p{3.4cm}p{12.4cm}", "When (IST) & what", [list(r) for r in rows], "Project timeline.", size="\\footnotesize"))


# =================================================================================================================
def repo_inventory():
    sec("Repository inventory and tests", "app:repo")
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "tgraph", "*.py")) + glob.glob(os.path.join(REPO, "scripts", "*.py")) +
                    glob.glob(os.path.join(REPO, "app", "*.py"))):
        if f.endswith("__init__.py") and "tgraph" not in f:
            continue
        doc = ast.get_docstring(ast.parse(open(f).read())) or ""
        rows.append([f"\\texttt{{{esc(os.path.relpath(f, REPO))}}}", esc(" ".join(doc.split(chr(10) * 2)[0].split()).replace("(spec ", "(build spec "))])
    for f in sorted(glob.glob(os.path.join(REPO, "jobs", "*.sh")) + glob.glob(os.path.join(REPO, "scripts", "*.sh"))):
        ls = [l_[1:].strip() for l_ in open(f) if l_.startswith("#") and not l_.startswith(("#!", "#SBATCH"))]
        rows.append([f"\\texttt{{{esc(os.path.relpath(f, REPO))}}}", esc(ls[0] if ls else "")])
    for f in sorted(glob.glob(os.path.join(REPO, "configs", "*"))):
        d = {"experiments.yaml": "every inference configuration", "graph_schema.json": "JSON schema of CG-JSON (constrained decoding)",
             "train_smoke.yaml": "tiny training run for the smoke tests"}.get(os.path.basename(f), "LoRA training configuration")
        rows.append([f"\\texttt{{{esc(os.path.relpath(f, REPO))}}}", d])
    S.append(longtable("p{4.6cm}p{11.3cm}", "File & purpose", rows,
                       "Every code and configuration file (descriptions are the files' own first comment lines).",
                       size="\\scriptsize"))
    rows = [("test\\_oracle.py", "the oracle reproduces all 5{,}000 dataset graphs exactly; nested scopes; a data-dependent \\texttt{if} raises; the sandbox reports errors; two-input models keep argument names"),
            ("test\\_schema.py", "CG-JSON round trip on all 5{,}000 graphs; the parser handles fences, think blocks, truncation and prose"),
            ("test\\_validate.py", "zero errors on every gold graph; each corruption is caught (deleted skip edge, back edge/cycle, undeclared layer, inserted dead code, changed conv shape, dropped output, missing layer and parse failure)"),
            ("test\\_metrics.py", "identity scores perfectly; corruptions lower the scores; renaming and permutation keep isomorphism; parse failure scores 0; AED is an upper bound of the true GED"),
            ("test\\_generate\\_logic.py", "with a fake model: few-shot prompts contain 3 examples, the retry turn contains the validator messages, a fixed graph is kept, self-consistency samples 5"),
            ("test\\_visualize.py", "both drawing back ends; diff and neuron-expansion drawings"),
            ("test\\_app\\_smoke.py", "the app handler: reference-only drawing, friendly errors for bad input, the untraceable message, every stored example renders")]
    S.append(table("p{3.6cm}p{12.2cm}", "Test file & what it checks", [list(r) for r in rows],
                   "The test suite (\\texttt{pytest -q}: 28 tests, all passing at the final commit).", size="\\footnotesize"))


# =================================================================================================================
def app_section():
    sec("The demo app", "app:app")
    par(r"""\textbf{What it shows.} Paste or upload an \texttt{nn.Module} (or pick one of the stored examples), choose a
model, and press \emph{Generate}. Tabs: the predicted graph, the torch.fx reference (if the code is traceable), the
colour-coded diff, the metrics and the 14 topological properties side by side, the validity report (pass/fail per
level, every message, retry rounds used), and both graphs as JSON (downloadable). Options: constrained decoding,
verify-and-retry, drawing small \texttt{Linear} layers as neurons, an input-shape override and the class name to use.
A \emph{Batch} tab scores a JSONL file of modules and returns a CSV of metrics. The pasted code is executed (in the
sandboxed oracle worker, 20\,s limit) to build the reference, so the app is a local tool, not a public service.

\textbf{Modes} (\texttt{scripts/run\_app.sh --mode \dots}):
\begin{itemize}[leftmargin=*]
\item \texttt{precomputed}: no model; the live torch.fx reference plus stored predictions, looked up by a hash of the
code: six configurations for every dataset and OOD sample when \texttt{results/raw} is present, otherwise the 45
predictions shipped in \texttt{app/precomputed.jsonl} (the dropdown examples). Runs on any laptop.
\item \texttt{cpu}: adds live generation with the Hugging Face backend (no constrained decoding). Measured on 16
server cores with FT-2B + synthetic: 4.4\,min for the first graph (including the model load), 2.5\,min per graph
afterwards, because Qwen3.5's linear-attention layers fall back to plain PyTorch without a GPU. Too slow for a live demo.
\item \texttt{gpu}: adds live generation with vLLM. The default model is FT-9B + synthetic; FT-9B, FT-2B (+ synthetic),
a base 2B with the adapter, and the base 9B with 3 examples are also offered. Each loaded model takes 45\% of the GPU, so
two fit on one L40S. With \texttt{--preload}, the default model is loaded and one warm-up request is sent before the
page opens (about 5 minutes); afterwards a graph takes about 9\,s.
\end{itemize}

\textbf{Running the live demo on Prajna} (from a terminal on the login node, ideally inside \texttt{tmux}):""")
    S.append(lst("cd ~/IE643_Project/topograph\n"
                 "srun -p interactive --qos=interactive --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \\\n"
                 "     --exclude=cn22-a40 bash jobs/app_gpu.sh\n"
                 "# wait for 'TopoGraph app ready'; the log prints the node name, then forward <node>:7860:\n"
                 "#   VS Code: Ports -> Forward a Port -> <node>:7860\n"
                 "#   laptop:  ssh -L 7860:<node>:7860 23b1835@login1.prajna.iitb.ac.in  ->  http://localhost:7860"))
    par(r"""To book a GPU ahead of time instead: \texttt{sbatch -p l40 --qos=l40 -c 8 --mem=64G -t 6:00:00 jobs/app\_gpu.sh}.
\textbf{Outside the cluster} (untested): FT-2B fits a free Colab T4; FT-9B needs about 19\,GB of GPU memory in bf16
(a Colab L4 is tight, an A100 comfortable). Ship the LoRA adapter (139\,MB for 2B, 326\,MB for 9B) and either merge it
with \texttt{scripts/merge\_lora.py} or load base model + adapter with PEFT; this was checked to give the identical graph.""")


def main():
    S.append("% generated by make_appendix_tex.py -- do not edit by hand\n\\setlength{\\tabcolsep}{3.5pt}")
    results()
    data_details()
    prompts_section()
    validator_metrics()
    training_inference()
    environment()
    decisions_issues_timeline()
    repo_inventory()
    app_section()
    with open(os.path.join(HERE, "appendix.tex"), "w") as f:
        f.write("\n\n".join(S) + "\n")
    print("wrote appendix.tex")


if __name__ == "__main__":
    main()
