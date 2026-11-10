"""Figures F1-F10 (spec §13) from results/tables + per-sample metrics. PNG + SVG.

Colour: one fixed categorical colour per config (colour follows the entity, never its rank); sequential blue for
heatmaps. Palette = dataviz reference palette (validated light-mode categorical order).
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.failures import LABELS  # noqa: E402
from tgraph.io import REPO  # noqa: E402
from tgraph.properties import ALL_PROPS  # noqa: E402

TAB = os.path.join(REPO, "results", "tables")
FIG = os.path.join(REPO, "results", "figures")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8984"
# fixed config -> slot (the 8 configs that appear together in the comparison figures)
CONFIG_COLOR = {"base9b_zs_c": SLOTS[0], "base9b_fs_c": SLOTS[1], "ft9b_c": SLOTS[2], "base2b_fs_c": SLOTS[3],
                "ft2b_c": SLOTS[4], "ft2b_aug_c": SLOTS[5], "ceil_fs_c": SLOTS[6], "ft9b_aug_c": SLOTS[7]}
LABEL = {"base9b_zs_c": "9B zero-shot", "base9b_fs_c": "9B 3-shot", "base9b_fs_u": "9B 3-shot (unconstr.)",
         "base2b_fs_c": "2B 3-shot", "ft2b_c": "2B fine-tuned", "ft9b_c": "9B fine-tuned", "ft9b_u": "9B FT (unconstr.)",
         "ft9b_c_retry": "9B FT + retry", "base9b_fs_c_retry": "9B 3-shot + retry", "coder7b_fs_c": "Coder-7B 3-shot",
         "ceil_fs_c": "27B 3-shot", "ceil_zs_c": "27B zero-shot", "ceil_zs_c_retry": "27B zero-shot + retry",
         "ceil_fs_c_retry": "27B 3-shot + retry", "base9b_think": "9B thinking (8k)", "base9b_think32k": "9B thinking (30k)",
         "base9b_think32k_c": "9B think-then-constrain", "ft9b_sc5": "9B FT self-consist.", "ft9b_sc5_retry": "9B FT SC + retry",
         "ft2b_c_retry": "2B FT + retry", "ft2b_aug_c": "2B FT + synthetic", "ft9b_aug_c": "9B FT + synthetic"}
MAIN_CONFIGS = list(CONFIG_COLOR)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def color(c):
    return CONFIG_COLOR.get(c, OTHER)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def present(df, configs=MAIN_CONFIGS):
    have = set(df["config"])
    return [c for c in configs if c in have]


def load_metrics():
    rows = []
    for f in glob.glob(os.path.join(REPO, "results", "raw", "*", "*.metrics.jsonl")):
        rows.append(pd.read_json(f, lines=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grouped_bars(ax, cats, configs, val, lo=None, hi=None):
    n = len(configs)
    w = 0.8 / max(1, n)
    x = np.arange(len(cats))
    for i, c in enumerate(configs):
        v = np.array([val(c, k) for k in cats], dtype=float)
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, v, width=w * 0.92, color=color(c), label=LABEL.get(c, c), edgecolor="white", linewidth=0.6)
        if lo is not None:
            l = np.array([lo(c, k) for k in cats], dtype=float)
            h = np.array([hi(c, k) for k in cats], dtype=float)
            ax.errorbar(pos, v, yerr=[v - l, h - v], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    ax.set_xticks(x)
    return x


def f1_headline(main):
    cfgs = present(main)
    if not cfgs:
        return
    metrics = ["fully_valid", "edge_f1", "exact_strict", "isomorphic", "graph_similarity", "pps", "shape_acc"]
    t = main.set_index("config")
    fig, ax = plt.subplots(figsize=(11, 4))
    grouped_bars(ax, metrics, cfgs, lambda c, m: 100 * t.loc[c, m],
                 lambda c, m: 100 * t.loc[c, m + "_lo"], lambda c, m: 100 * t.loc[c, m + "_hi"])
    ax.set_xticklabels([m.replace("_", " ") for m in metrics])
    ax.set_ylabel("% of test samples / mean score")
    ax.set_ylim(0, 105)
    ax.set_title("F1. Headline metrics on the official test set (95% bootstrap CI)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "F1_headline")


def f2_tier(df):
    t = df[df.eval_set == "test"]
    cfgs = present(t)
    if not cfgs:
        return
    tiers = ["L0", "L1", "L2", "L3", "L4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m, title in ((axes[0], "edge_f1", "Edge F1"), (axes[1], "exact_strict", "Exact match")):
        g = t.groupby(["config", "tier"])[m].mean()
        for c in cfgs:
            ys = [100 * g.get((c, k), np.nan) for k in tiers]
            ax.plot(tiers, ys, color=color(c), lw=2, marker="o", ms=5, label=LABEL.get(c, c))
        ax.set_title(title, loc="left")
        ax.set_xlabel("complexity tier")
    axes[0].set_ylabel("% (official test)")
    axes[0].set_ylim(0, 105)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("F2. Accuracy vs topological complexity", x=0.01, ha="left")
    save(fig, "F2_by_tier")


def f3_size(df):
    t = df[df.eval_set == "test"]
    cfgs = present(t)
    if not cfgs:
        return
    bins = [b for b in ["<=10", "11-15", "16-20", ">20"] if b in set(t.size_bin)]
    g = t.groupby(["config", "size_bin"]).graph_similarity.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in cfgs:
        ax.plot(bins, [100 * g.get((c, b), np.nan) for b in bins], color=color(c), lw=2, marker="o", ms=5,
                label=LABEL.get(c, c))
    ax.set_xlabel("reference graph size (nodes)")
    ax.set_ylabel("graph similarity (1 - normalised AED), %")
    ax.set_title("F3. Graph similarity vs graph size", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "F3_by_size")


def f4_seen(df):
    p = os.path.join(TAB, "seen_vs_unseen.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    cfgs = present(s)
    slices = [x for x in ["test_seen", "test_unseen", "novel_topo", "ood", "ood2"] if x in set(s["slice"])]
    if not cfgs or not slices:
        return
    t = s.set_index(["config", "slice"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, m, title in ((axes[0], "edge_f1", "Edge F1"), (axes[1], "exact_strict", "Exact match")):
        grouped_bars(ax, slices, cfgs, lambda c, k: 100 * t[m].get((c, k), np.nan))
        names = {"test_seen": "test\nseen", "test_unseen": "test\nheld-out", "novel_topo": "novel\ntopo",
                 "ood": "OOD-1", "ood2": "OOD-2"}
        ax.set_xticklabels([names[x] for x in slices])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("%")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("F4. Seen vs unseen topologies (memorisation vs generalisation)", x=0.01, ha="left")
    save(fig, "F4_seen_vs_unseen")


def f5_robust(df):
    t = df[df.eval_set.isin(["test", "test_anon"])]
    cfgs = present(t)
    if not cfgs:
        return
    augs = ["none", "aug_variant_1", "aug_variant_2", "aug_variant_3"]
    g = t[t.eval_set == "test"].groupby(["config", "augmentation_type"]).edge_f1.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
    grouped_bars(axes[0], augs, cfgs, lambda c, a: 100 * g.get((c, a), np.nan))
    axes[0].set_xticklabels(["original", "arg renamed", "dead code", "rename+dead+shuffle"])
    axes[0].set_title("by augmentation (test)", loc="left")
    axes[0].set_ylabel("edge F1, %")
    ga = t.groupby(["config", "eval_set"]).edge_f1.mean()
    c2 = [c for c in cfgs if (c, "test_anon") in ga.index]
    if c2:
        grouped_bars(axes[1], ["test", "test_anon"], c2, lambda c, e: 100 * ga.get((c, e), np.nan))
        axes[1].set_xticklabels(["original class name", "class -> Net"])
    axes[1].set_title("anonymised class name", loc="left")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("F5. Robustness to surface changes", x=0.01, ha="left")
    save(fig, "F5_robustness")


def f6_validity():
    p = os.path.join(TAB, "validity.csv")
    if not os.path.exists(p):
        return
    v = pd.read_csv(p)
    v = v[v.eval_set == "test"]
    pairs = [("base9b_fs_c", "base9b_fs_u"), ("ft9b_c", "ft9b_u")]
    rows = [(a, b) for a, b in pairs if a in set(v.config) and b in set(v.config)]
    levels = ["parse_ok", "struct_valid", "arity_valid", "code_consistent", "shape_consistent", "fully_valid"]
    r = os.path.join(TAB, "retry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    if rows:
        cfgs = [c for ab in rows for c in ab]
        t = v.set_index("config")
        x = np.arange(len(levels))
        w = 0.8 / len(cfgs)
        for i, c in enumerate(cfgs):
            constrained = c.endswith("_c")
            base = CONFIG_COLOR.get(c.replace("_u", "_c"), OTHER)
            axes[0].bar(x - 0.4 + w * (i + 0.5), [100 * t.loc[c, l] for l in levels], width=w * 0.92, color=base,
                        hatch=None if constrained else "///", edgecolor="white", label=LABEL.get(c, c))
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([l.replace("_", " ") for l in levels])
        axes[0].set_ylabel("% of test samples")
        axes[0].legend(ncol=2, loc="lower left")
    axes[0].set_title("constrained (solid) vs unconstrained (hatched)", loc="left")
    if os.path.exists(r):
        rt = pd.read_csv(r)
        lab = [f"{LABEL.get(c, c)}\n{e}" for c, e in zip(rt.config, rt.eval_set)]
        y = np.arange(len(rt))
        axes[1].barh(y - 0.2, 100 * rt.initial_fully_valid, height=0.38, color="#86b6ef", label="before retry")
        axes[1].barh(y + 0.2, 100 * rt.final_fully_valid, height=0.38, color="#1c5cab", label="after retry")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(lab, fontsize=8)
        axes[1].set_xlabel("fully valid, %")
        axes[1].legend(loc="lower right")
    axes[1].set_title("verify-and-retry", loc="left")
    fig.suptitle("F6. Validity", x=0.01, ha="left")
    save(fig, "F6_validity")


def f7_failures(df):
    t = df[df.eval_set == "test"]
    cfgs = present(t, list(LABEL))
    if not cfgs:
        return
    ex = t.explode("failures")
    cnt = ex.groupby(["config", "failures"]).size().unstack(fill_value=0)
    labels = [l for l in LABELS if l in cnt.columns]
    cnt = cnt.reindex(cfgs).fillna(0)
    n = t.groupby("config").size().reindex(cfgs)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    # categorical labels: use the categorical palette for the first 8, grey family after
    colors = SLOTS + ["#b5b4ad", "#8a8984", "#52514e"]
    bottom = np.zeros(len(cfgs))
    for i, l in enumerate(labels):
        vals = 100 * cnt[l].values / n.values
        ax.bar(range(len(cfgs)), vals, bottom=bottom, color=colors[i % len(colors)], edgecolor="white", linewidth=1,
               label=l.replace("_", " "))
        bottom += vals
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([LABEL.get(c, c) for c in cfgs], rotation=25, ha="right")
    ax.set_ylabel("failure labels per 100 test samples")
    ax.set_title("F7. Failure taxonomy (a sample can carry several labels)", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "F7_failures")


def f8_ged():
    p = os.path.join(TAB, "ged_subset.csv")
    if not os.path.exists(p):
        return
    g = pd.read_csv(p).dropna(subset=["ged"])
    from scipy.stats import spearmanr
    rho = spearmanr(g.aed, g.ged).correlation if len(g) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(g.ged, g.aed, s=22, color=SLOTS[0], alpha=0.6, edgecolor="white", linewidth=0.5)
    m = max(g.aed.max(), g.ged.max()) + 1
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--", label="AED = GED")
    ax.set_xlabel("true GED (best within 2 s)")
    ax.set_ylabel("aligned edit distance (AED)")
    ax.set_title(f"F8. AED vs GED (Spearman ρ = {rho:.2f}, n = {len(g)})", loc="left")
    ax.legend(loc="upper left")
    save(fig, "F8_aed_vs_ged")


def f9_loss():
    fig, ax = plt.subplots(figsize=(7, 4))
    found = False
    for tag, c in (("train_2b", CONFIG_COLOR["ft2b_c"]), ("train_9b", CONFIG_COLOR["ft9b_c"])):
        p = os.path.join(REPO, "runs", tag, "log_history.csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        name = "2B" if "2b" in tag else "9B"
        if "loss" in h:
            tr = h.dropna(subset=["loss"])
            ax.plot(tr.step, tr.loss, color=c, lw=1.5, label=f"{name} train")
        if "eval_loss" in h:
            ev = h.dropna(subset=["eval_loss"])
            ax.plot(ev.step, ev.eval_loss, color=c, lw=2, ls="--", marker="o", ms=4, label=f"{name} val")
        found = True
    if not found:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy on completion tokens (log)")
    ax.set_title("F9. Fine-tuning loss", loc="left")
    ax.legend()
    save(fig, "F9_loss")


def f10_properties():
    p = os.path.join(TAB, "properties.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    t = t[t.eval_set == "test"].set_index("config")
    cfgs = [c for c in LABEL if c in t.index]
    if not cfgs:
        return
    cols = [f"prop_{k}" for k in ALL_PROPS]
    M = 100 * t.loc[cfgs, cols].values
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(cfgs) + 1.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[5:].replace("_", " ") for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([LABEL.get(c, c) for c in cfgs])
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 55 else INK)
    fig.colorbar(im, ax=ax, label="% of test samples with the property preserved")
    ax.set_title("F10. Topological property preservation", loc="left")
    save(fig, "F10_properties")


def main():
    df = load_metrics()
    mp = os.path.join(TAB, "main.csv")
    if os.path.exists(mp):
        f1_headline(pd.read_csv(mp))
    if not df.empty:
        f2_tier(df)
        f3_size(df)
        f5_robust(df)
        f7_failures(df)
    f4_seen(df)
    f6_validity()
    f8_ged()
    f9_loss()
    f10_properties()


if __name__ == "__main__":
    main()
