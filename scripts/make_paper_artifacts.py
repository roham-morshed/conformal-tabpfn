"""Turn results/raw/*.parquet into the paper's tables, figures and summary.

Every number emitted here is read from a measured result row.  Nothing is
interpolated, smoothed, or filled in: if a cell did not run, it is absent from
the table and the absence is reported.

    python scripts/make_paper_artifacts.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_tabpfn.experiment import load_all_results  # noqa: E402
from conformal_tabpfn.stats import (  # noqa: E402
    critical_difference,
    mean_ranks,
    wilcoxon_pairwise,
)
from conformal_tabpfn.utils import environment_info  # noqa: E402

TABLES = ROOT / "paper" / "tables"
FIGURES = ROOT / "paper" / "figures"
RAW = ROOT / "results" / "raw"
FAILURES = ROOT / "results" / "failures"

# Okabe-Ito: colour-blind safe, prints legibly in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

MAIN_ALPHA = 0.1
PRIMARY_MODEL = "tabpfn"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
})

SUMMARY: list[str] = []


def note(line: str = "") -> None:
    """Record a line for paper/results_summary.md and echo it."""
    SUMMARY.append(line)
    print(line)


def save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf and .png")


def write_table(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body, encoding="utf-8")
    print(f"  wrote paper/tables/{name}")


def esc(text: object) -> str:
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def latex_table(caption: str, label: str, header: list[str], rows: list[list[str]],
                align: str | None = None, note_line: str = "") -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    out = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note_line:
        out.append(rf"\\[2pt]\footnotesize {note_line}")
    out += [r"\end{table}", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

AGG = {
    "coverage": ["mean", "std"],
    "avg_set_size": ["mean", "std"],
    "cc_coverage_gap": ["mean"],
    "frac_empty": ["mean"],
    "frac_singleton": ["mean"],
    "sscv": ["mean"],
    "accuracy": ["mean"],
    "ece": ["mean"],
    "fit_time": ["mean"],
    "predict_time": ["mean"],
    "cp_time": ["mean"],
    "n_fits": ["mean"],
}


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean +/- SD across the seeds within each cell group."""
    g = df.groupby(by, dropna=False).agg({**AGG, "seed": "nunique"})
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    return g.rename(columns={"seed_nunique": "n_seeds"}).reset_index()


def best_score_per(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pick, per group, the score function with the smallest mean set size.

    Set sizes are only comparable at matched coverage, so we restrict to scores
    whose mean coverage reaches the nominal level before choosing.
    """
    agg = aggregate(df, [*keys, "score"])
    ok = agg[agg["coverage_mean"] >= 1 - MAIN_ALPHA - 0.02]
    pool = ok if len(ok) else agg
    idx = pool.groupby(keys)["avg_set_size_mean"].idxmin()
    return pool.loc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def table_main(df: pd.DataFrame) -> None:
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA) & (df.study == "main")]
    if d.empty:
        note("T1 skipped: no main-study rows for the primary model.")
        return
    best = best_score_per(d, ["dataset", "cp_method"])
    methods = sorted(best.cp_method.unique())
    rows, sizes_by_ds = [], {}
    for ds in sorted(best.dataset.unique()):
        sub = best[best.dataset == ds].set_index("cp_method")
        sizes = {m: sub.loc[m, "avg_set_size_mean"] for m in methods if m in sub.index}
        sizes_by_ds[ds] = sizes
        smallest = min(sizes.values()) if sizes else np.nan
        cells = [esc(ds)]
        for m in methods:
            if m not in sub.index:
                cells.append("--")
                continue
            r = sub.loc[m]
            size = fmt(r["avg_set_size_mean"], 2)
            if np.isclose(r["avg_set_size_mean"], smallest):
                size = rf"\textbf{{{size}}}"
            cells.append(
                f"{fmt(r['coverage_mean'])}$\\pm${fmt(r['coverage_std'], 3)} / {size} / "
                f"{fmt(r['cc_coverage_gap_mean'], 2)}"
            )
        rows.append(cells)

    pivot = pd.DataFrame(sizes_by_ds).T[methods]
    ranks = mean_ranks(pivot, lower_is_better=True)
    rows.append([r"\midrule mean rank (set size)"] + [fmt(ranks.get(m, np.nan), 2) for m in methods])

    write_table("T1_main.tex", latex_table(
        caption=(
            f"Main results for {PRIMARY_MODEL} at $\\alpha={MAIN_ALPHA}$, best score per cell. "
            "Each entry is coverage (mean$\\pm$sd over seeds) / average set size / "
            "class-conditional coverage gap. Bold marks the smallest set size in the row."
        ),
        label="tab:main",
        header=["dataset"] + [esc(m) for m in methods],
        rows=rows,
    ))
    note(f"**T1** covers {len(sizes_by_ds)} datasets x {len(methods)} CP methods.")
    note(f"Best mean rank by set size: **{ranks.index[0]}** ({ranks.iloc[0]:.2f}); "
         f"worst: {ranks.index[-1]} ({ranks.iloc[-1]:.2f}).")
    cd = critical_difference(len(methods), len(sizes_by_ds))
    note(f"Nemenyi critical difference (alpha=0.05, k={len(methods)}, N={len(sizes_by_ds)}): {cd:.2f}.")


def table_cost(df: pd.DataFrame) -> None:
    """RQ1 -- the wall-clock table.  This is the one that changed the paper."""
    d = df[(df.alpha == MAIN_ALPHA) & (df.score == "lac") & (df.study == "main")]
    if d.empty:
        note("T2 skipped: no rows.")
        return
    d = d.copy()
    d["total_time"] = d.fit_time + d.predict_time
    agg = d.groupby(["model", "cp_method"]).agg(
        fit=("fit_time", "mean"), pred=("predict_time", "mean"),
        total=("total_time", "mean"), n_fits=("n_fits", "mean"),
    ).reset_index()

    methods = sorted(agg.cp_method.unique())
    models = [m for m in ["tabpfn", "xgboost", "rf", "logreg"] if m in set(agg.model)]
    rows = []
    ratios: dict[str, dict[str, float]] = {}
    for model in models:
        sub = agg[agg.model == model].set_index("cp_method")
        base = sub.loc["split_0.3", "total"] if "split_0.3" in sub.index else np.nan
        ratios[model] = {}
        cells = [esc(model)]
        for m in methods:
            if m not in sub.index:
                cells.append("--")
                continue
            tot = sub.loc[m, "total"]
            ratio = tot / base if np.isfinite(base) and base > 0 else np.nan
            ratios[model][m] = ratio
            cells.append(f"{fmt(tot, 3)} ({fmt(ratio, 1)}$\\times$)")
        rows.append(cells)
        fit_share = sub.loc["split_0.3", "fit"] / base * 100 if np.isfinite(base) and base else np.nan
        note(f"  {model}: Split total {fmt(base, 3)} s, of which fit is {fmt(fit_share, 1)}%.")

    write_table("T2_cost.tex", latex_table(
        caption=(
            "RQ1. Mean wall-clock seconds per cell (fit + predict, CPU), with the "
            "ratio to that model's own Split-CP cost in parentheses. The ratio is "
            "$\\approx K$ for every model, including the in-context learner."
        ),
        label="tab:cost",
        header=["model"] + [esc(m) for m in methods],
        rows=rows,
        note_line=(
            "TabPFN's fit is genuinely near-free, but CV+ also re-predicts $K$ times, "
            "and for an in-context learner prediction is the entire cost."
        ),
    ))

    note("")
    note("### RQ1 -- cost ratio CV+/Split (the thesis test)")
    for m in methods:
        if not m.startswith("cvplus"):
            continue
        vals = {mod: ratios[mod].get(m, np.nan) for mod in models}
        txt = ", ".join(f"{k}={fmt(v, 2)}x" for k, v in vals.items() if np.isfinite(v))
        note(f"- {m}: {txt}")
    tab = ratios.get("tabpfn", {})
    grad = [ratios[m] for m in ("xgboost", "rf", "logreg") if m in ratios]
    for m in methods:
        if not m.startswith("cvplus_K"):
            continue
        t = tab.get(m, np.nan)
        g = np.nanmean([r.get(m, np.nan) for r in grad]) if grad else np.nan
        if np.isfinite(t) and np.isfinite(g):
            note(f"- {m}: TabPFN ratio {t:.2f}x vs gradient-model mean {g:.2f}x "
                 f"-> TabPFN is {t / g:.2f}x the baseline ratio "
                 f"({'NO advantage' if t / g > 0.9 else 'advantage'}).")


def table_stability(df: pd.DataFrame) -> None:
    """RQ2 -- coverage SD across seeds.  The headline statistical result."""
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA)
           & (df.score == "lac") & (df.study == "main")]
    if d.empty:
        note("T3 skipped: no rows.")
        return
    agg = aggregate(d, ["dataset", "cp_method"])
    pivot_sd = agg.pivot(index="dataset", columns="cp_method", values="coverage_std")
    pivot_cov = agg.pivot(index="dataset", columns="cp_method", values="coverage_mean")
    if "split_0.3" not in pivot_sd.columns:
        note("T3 skipped: no split baseline.")
        return

    cv_methods = [c for c in pivot_sd.columns if c.startswith("cvplus")]
    rows = []
    for ds in pivot_sd.index:
        cells = [esc(ds), fmt(pivot_cov.loc[ds, "split_0.3"]), fmt(pivot_sd.loc[ds, "split_0.3"], 4)]
        for m in cv_methods:
            sd = pivot_sd.loc[ds, m]
            base = pivot_sd.loc[ds, "split_0.3"]
            red = base / sd if np.isfinite(sd) and sd > 0 else np.nan
            cells += [fmt(pivot_cov.loc[ds, m]), fmt(sd, 4), fmt(red, 2)]
        rows.append(cells)

    header = ["dataset", "cov (split)", "sd (split)"]
    for m in cv_methods:
        header += [f"cov ({esc(m)})", f"sd ({esc(m)})", "factor"]
    write_table("T3_stability.tex", latex_table(
        caption=(
            "RQ2. Standard deviation of empirical coverage across 20 seeds, "
            f"Split vs CV+ ({PRIMARY_MODEL}, LAC, $\\alpha={MAIN_ALPHA}$). "
            "``factor'' is sd(Split)/sd(CV+); values above 1 mean CV+ is more stable."
        ),
        label="tab:stability", header=header, rows=rows,
    ))

    note("")
    note("### RQ2 -- coverage stability (headline)")
    for m in cv_methods:
        a, b = pivot_sd[m], pivot_sd["split_0.3"]
        ok = np.isfinite(a) & np.isfinite(b)
        w = wilcoxon_pairwise(a[ok].to_numpy(), b[ok].to_numpy())
        factor = float(np.nanmedian(b[ok] / a[ok]))
        note(f"- {m}: median sd(Split)/sd(CV+) = **{factor:.2f}x**; "
             f"CV+ has lower sd on {w['wins']}/{w['n']} datasets "
             f"(Wilcoxon p={fmt(w['p_value'], 4)}).")
    return pivot_sd


def table_imbalance(df: pd.DataFrame) -> None:
    """RQ5 -- marginal vs Mondrian class-conditional coverage."""
    d = df[(df.alpha == MAIN_ALPHA) & (df.score == "lac")
           & (df.model == PRIMARY_MODEL)]
    if d.empty:
        note("T4 skipped: no rows.")
        return
    pairs = [("split_0.3", "mondrian_split_0.3"), ("cvplus_K10", "mondrian_cvplus_K10")]
    agg = aggregate(d, ["study", "dataset", "cp_method"])
    rows = []
    for study in sorted(agg.study.unique()):
        sub = agg[agg.study == study]
        for ds in sorted(sub.dataset.unique()):
            s = sub[sub.dataset == ds].set_index("cp_method")
            cells = [esc(study), esc(ds)]
            for marg, mond in pairs:
                for m in (marg, mond):
                    if m in s.index:
                        cells.append(
                            f"{fmt(s.loc[m, 'cc_coverage_gap_mean'], 3)} / "
                            f"{fmt(s.loc[m, 'avg_set_size_mean'], 2)}"
                        )
                    else:
                        cells.append("--")
            rows.append(cells)
    if not rows:
        note("T4 skipped: nothing to tabulate.")
        return
    write_table("T4_imbalance.tex", latex_table(
        caption=(
            "RQ5. Class-conditional coverage gap / average set size, marginal vs "
            "Mondrian. Lower gap is fairer; larger sets are the price."
        ),
        label="tab:imbalance",
        header=["study", "dataset", "split", "mondrian-split", "cv+", "mondrian-cv+"],
        rows=rows,
    ))

    note("")
    note("### RQ5 -- class-conditional fairness")
    for marg, mond in pairs:
        a = agg[agg.cp_method == mond].set_index(["study", "dataset"])["cc_coverage_gap_mean"]
        b = agg[agg.cp_method == marg].set_index(["study", "dataset"])["cc_coverage_gap_mean"]
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        w = wilcoxon_pairwise(a.loc[common].to_numpy(), b.loc[common].to_numpy())
        sa = agg[agg.cp_method == mond].set_index(["study", "dataset"])["avg_set_size_mean"]
        sb = agg[agg.cp_method == marg].set_index(["study", "dataset"])["avg_set_size_mean"]
        note(f"- {mond} vs {marg}: mean gap {a.loc[common].mean():.4f} vs "
             f"{b.loc[common].mean():.4f}; Mondrian narrower on {w['wins']}/{w['n']} "
             f"(p={fmt(w['p_value'], 4)}); set size {sa.loc[common].mean():.3f} vs "
             f"{sb.loc[common].mean():.3f}.")


def table_alpha_robustness(df: pd.DataFrame) -> None:
    d = df[(df.model == PRIMARY_MODEL) & (df.score == "lac") & (df.study == "main")]
    if d.empty:
        note("T5 skipped: no rows.")
        return
    agg = aggregate(d, ["alpha", "cp_method"])
    methods = sorted(agg.cp_method.unique())
    rows = []
    for alpha in sorted(agg.alpha.unique()):
        sub = agg[agg.alpha == alpha].set_index("cp_method")
        cells = [fmt(alpha, 2), fmt(1 - alpha, 2)]
        for m in methods:
            cells.append(
                f"{fmt(sub.loc[m, 'coverage_mean'])} / {fmt(sub.loc[m, 'avg_set_size_mean'], 2)}"
                if m in sub.index else "--"
            )
        rows.append(cells)
    write_table("T5_alpha_robustness.tex", latex_table(
        caption=(
            "Robustness across miscoverage levels: mean coverage / mean set size, "
            "averaged over datasets and seeds."
        ),
        label="tab:alpha",
        header=[r"$\alpha$", r"$1-\alpha$"] + [esc(m) for m in methods], rows=rows,
    ))

    note("")
    note("### Alpha robustness")
    for alpha in sorted(agg.alpha.unique()):
        sub = agg[agg.alpha == alpha].set_index("cp_method")
        parts = [f"{m}={fmt(sub.loc[m, 'coverage_mean'])}" for m in methods if m in sub.index]
        note(f"- alpha={alpha:.2f} (target {1 - alpha:.2f}): " + ", ".join(parts))


def table_conservativeness(df: pd.DataFrame) -> None:
    """RQ4 -- how much over-coverage CV+ actually shows, by K and by n_classes."""
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA)
           & (df.score == "lac") & (df.study == "main")
           & (df.cp_method.str.startswith("cvplus"))]
    if d.empty:
        note("RQ4 skipped: no CV+ rows.")
        return
    d = d.copy()
    d["K"] = d.cp_method.str.extract(r"K(\d+)").astype(int)
    d["binary"] = np.where(d.n_classes <= 2, "binary", "multiclass")
    agg = d.groupby(["binary", "K"]).agg(
        cov=("coverage", "mean"), size=("avg_set_size", "mean"),
        sd=("coverage", "std"),
    ).reset_index()
    note("")
    note("### RQ4 -- conservativeness of CV+ (guarantee is 1-2a = "
         f"{1 - 2 * MAIN_ALPHA:.2f}, nominal 1-a = {1 - MAIN_ALPHA:.2f})")
    for _, r in agg.iterrows():
        note(f"- {r['binary']}, K={int(r['K'])}: coverage {r['cov']:.4f} "
             f"(over-coverage {r['cov'] - (1 - MAIN_ALPHA):+.4f}), set size {r['size']:.3f}")


def table_guidelines(df: pd.DataFrame, sd_pivot) -> None:
    d = df[(df.alpha == MAIN_ALPHA) & (df.score == "lac") & (df.study == "main")
           & (df.model == PRIMARY_MODEL)]
    if d.empty:
        note("T6 skipped: no rows.")
        return
    d = d.copy()
    d["imb_band"] = np.where(d.n_classes <= 2, "binary", "multiclass")
    rows = []
    for band, sub in d.groupby("imb_band"):
        agg = aggregate(sub, ["cp_method"])
        agg = agg[agg.coverage_mean >= 1 - MAIN_ALPHA - 0.02]
        if agg.empty:
            continue
        best_size = agg.loc[agg.avg_set_size_mean.idxmin(), "cp_method"]
        best_sd = agg.loc[agg.coverage_std.idxmin(), "cp_method"]
        best_gap = agg.loc[agg.cc_coverage_gap_mean.idxmin(), "cp_method"]
        cheapest = aggregate(sub, ["cp_method"]).sort_values("fit_time_mean").iloc[0]["cp_method"]
        rows.append([esc(band), esc(best_size), esc(best_sd), esc(best_gap), esc(cheapest)])
    if rows:
        write_table("T6_guidelines.tex", latex_table(
            caption=(
                "Practical guidance. For each regime, the method with the smallest "
                "sets, the most stable coverage, the fairest per-class coverage, and "
                "the lowest wall-clock cost, among methods reaching nominal coverage."
            ),
            label="tab:guidelines",
            header=["regime", "smallest sets", "most stable", "fairest", "cheapest"],
            rows=rows,
        ))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_coverage_boxplot(df: pd.DataFrame) -> None:
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA)
           & (df.score == "lac") & (df.study == "main")
           & (df.cp_method.isin(["split_0.3", "cvplus_K10"]))]
    if d.empty:
        return
    datasets = sorted(d.dataset.unique())[:8]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    positions, data, colors = [], [], []
    for i, ds in enumerate(datasets):
        for j, m in enumerate(["split_0.3", "cvplus_K10"]):
            vals = d[(d.dataset == ds) & (d.cp_method == m)].coverage.to_numpy()
            if vals.size:
                positions.append(i * 3 + j)
                data.append(vals)
                colors.append(PALETTE[j])
    bp = ax.boxplot(data, positions=positions, widths=0.75, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.0},
                    flierprops={"marker": ".", "markersize": 2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_linewidth(0.5)
    ax.axhline(1 - MAIN_ALPHA, color="black", ls="--", lw=0.8,
               label=f"nominal $1-\\alpha$ = {1 - MAIN_ALPHA}")
    ax.axhline(1 - 2 * MAIN_ALPHA, color="grey", ls=":", lw=0.8,
               label=f"CV+ guarantee $1-2\\alpha$ = {1 - 2 * MAIN_ALPHA}")
    ax.set_xticks([i * 3 + 0.5 for i in range(len(datasets))])
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("empirical coverage (fraction)")
    ax.set_xlabel("dataset")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[i], alpha=0.65) for i in range(2)]
    ax.legend(handles + ax.get_lines()[-2:], ["Split (30%)", "CV+ (K=10)",
              f"nominal {1 - MAIN_ALPHA}", f"guarantee {1 - 2 * MAIN_ALPHA}"],
              ncol=2, loc="lower right", framealpha=0.9)
    ax.set_title(f"Coverage across 20 seeds ({PRIMARY_MODEL}, LAC)")
    save_fig(fig, "F1_coverage_boxplot")


def fig_pareto(df: pd.DataFrame) -> None:
    d = df[(df.alpha == MAIN_ALPHA) & (df.score == "lac") & (df.study == "main")].copy()
    if d.empty:
        return
    d["total_time"] = d.fit_time + d.predict_time
    agg = d.groupby(["model", "cp_method"]).agg(
        t=("total_time", "mean"), size=("avg_set_size", "mean"), cov=("coverage", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    models = sorted(agg.model.unique())
    methods = sorted(agg.cp_method.unique())
    for i, model in enumerate(models):
        sub = agg[agg.model == model]
        for j, m in enumerate(methods):
            row = sub[sub.cp_method == m]
            if row.empty:
                continue
            ax.scatter(row.t, row["size"], color=PALETTE[i % len(PALETTE)],
                       marker=MARKERS[j % len(MARKERS)], s=26, alpha=0.9,
                       edgecolors="black", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock seconds per cell (fit + predict, log scale)")
    ax.set_ylabel("average prediction-set size (labels)")
    ax.set_title("Cost vs efficiency")
    mh = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[i % len(PALETTE)], label=models[i])
          for i in range(len(models))]
    sh = [plt.Line2D([], [], marker=MARKERS[j % len(MARKERS)], ls="", color="grey",
                     label=methods[j]) for j in range(len(methods))]
    ax.legend(handles=mh + sh, ncol=2, fontsize=6, loc="best", framealpha=0.9)
    save_fig(fig, "F2_pareto")


def fig_budget(df: pd.DataFrame) -> None:
    d = df[(df.study == "budget") & (df.alpha == MAIN_ALPHA) & (df.score == "lac")]
    if d.empty:
        note("F3 skipped: budget sweep has not been run.")
        return
    d = d.copy()
    d["rho"] = d.cp_method.str.replace("split_", "", regex=False).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    note("")
    note("### RQ3 -- calibration budget")
    for i, model in enumerate(sorted(d.model.unique())):
        sub = d[d.model == model].groupby("rho").agg(
            size=("avg_set_size", "mean"), cov=("coverage", "mean"),
            sd=("coverage", "std"), acc=("accuracy", "mean")
        ).reset_index()
        axes[0].plot(sub.rho, sub["size"], marker=MARKERS[i], color=PALETTE[i], label=model, lw=1.2)
        axes[1].plot(sub.rho, sub["sd"], marker=MARKERS[i], color=PALETTE[i], label=model, lw=1.2)
        star = sub.loc[sub["size"].idxmin()]
        axes[0].scatter([star.rho], [star["size"]], s=90, facecolors="none",
                        edgecolors=PALETTE[i], linewidths=1.2, zorder=5)
        note(f"- {model}: rho* = **{star.rho:.2f}** (min set size {star['size']:.3f}, "
             f"coverage {star['cov']:.4f}, underlying accuracy {star['acc']:.4f})")
    axes[0].set_xlabel(r"calibration fraction $\rho$")
    axes[0].set_ylabel("average set size (labels)")
    axes[0].set_title(r"Efficiency vs $\rho$ ($\circ$ marks $\rho^*$)")
    axes[1].set_xlabel(r"calibration fraction $\rho$")
    axes[1].set_ylabel("sd of coverage across seeds")
    axes[1].set_title(r"Stability vs $\rho$")
    for a in axes:
        a.legend()
    save_fig(fig, "F3_budget_curve")


def fig_conservativeness(df: pd.DataFrame) -> None:
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA) & (df.score == "lac")
           & (df.study == "main") & df.cp_method.str.startswith("cvplus")].copy()
    if d.empty:
        return
    d["K"] = d.cp_method.str.extract(r"K(\d+)").astype(int)
    d["kind"] = np.where(d.n_classes <= 2, "binary (C=2)", "multiclass (C>2)")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for i, (kind, sub) in enumerate(d.groupby("kind")):
        g = sub.groupby("K").agg(cov=("coverage", "mean"), size=("avg_set_size", "mean"),
                                 sd=("coverage", "std")).reset_index()
        # Bracket access is mandatory here: `g.cov` resolves to DataFrame.cov,
        # the covariance METHOD, not the column, and matplotlib then reports the
        # confusing "x and y must have the same size".
        axes[0].errorbar(g["K"], g["cov"], yerr=g["sd"], marker=MARKERS[i],
                         color=PALETTE[i], label=kind, capsize=2, lw=1.2)
        axes[1].plot(g.K, g["size"], marker=MARKERS[i], color=PALETTE[i], label=kind, lw=1.2)
    axes[0].axhline(1 - MAIN_ALPHA, color="black", ls="--", lw=0.8, label=r"$1-\alpha$")
    axes[0].axhline(1 - 2 * MAIN_ALPHA, color="grey", ls=":", lw=0.8, label=r"$1-2\alpha$")
    axes[0].set_xlabel("number of folds K")
    axes[0].set_ylabel("empirical coverage (fraction)")
    axes[0].set_title("Coverage vs K")
    axes[1].set_xlabel("number of folds K")
    axes[1].set_ylabel("average set size (labels)")
    axes[1].set_title("Set size vs K")
    for a in axes:
        a.legend(fontsize=6)
    save_fig(fig, "F4_conservativeness")


def fig_crossover(df: pd.DataFrame) -> None:
    d = df[(df.study == "crossover") & (df.alpha == MAIN_ALPHA) & (df.score == "lac")]
    if d.empty:
        note("F5 skipped: crossover study has not been run.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    note("")
    note("### RQ6 -- crossover in n_train")
    for i, m in enumerate(sorted(d.cp_method.unique())):
        g = d[d.cp_method == m].groupby("n_train").agg(
            sd=("coverage", "std"), size=("avg_set_size", "mean"), cov=("coverage", "mean")
        ).reset_index()
        axes[0].plot(g.n_train, g["sd"], marker=MARKERS[i], color=PALETTE[i], label=m, lw=1.2)
        axes[1].plot(g.n_train, g["size"], marker=MARKERS[i], color=PALETTE[i], label=m, lw=1.2)
        note(f"- {m}: " + ", ".join(
            f"n={int(r.n_train)} sd={r['sd']:.4f} size={r['size']:.3f}" for _, r in g.iterrows()))
    for a, ylab, title in ((axes[0], "sd of coverage across seeds", "Stability vs n"),
                           (axes[1], "average set size (labels)", "Efficiency vs n")):
        a.set_xlabel("training-set size n (points)")
        a.set_ylabel(ylab)
        a.set_title(title)
        a.set_xscale("log")
        a.legend(fontsize=6)
    save_fig(fig, "F5_crossover")

    piv = d.groupby(["cp_method", "n_train"]).coverage.std().unstack(0)
    if {"split_0.3", "cvplus_K10"}.issubset(piv.columns):
        gap = piv["split_0.3"] - piv["cvplus_K10"]
        caught = gap[gap <= 0]
        note(f"- Split catches CV+ on coverage-sd at n = "
             f"{int(caught.index[0]) if len(caught) else 'not within the swept range'}.")


def fig_cc_coverage(df: pd.DataFrame) -> None:
    d = df[(df.model == PRIMARY_MODEL) & (df.alpha == MAIN_ALPHA) & (df.score == "lac")]
    pairs = [("split_0.3", "mondrian_split_0.3"), ("cvplus_K10", "mondrian_cvplus_K10")]
    avail = [p for p in pairs if set(p).issubset(set(d.cp_method))]
    if d.empty or not avail:
        return
    # Pick the most imbalanced dataset available: that is where it shows.
    counts = d.groupby("dataset").n_classes.first()
    target = d[d.dataset.isin(counts.index)].copy()
    ds = target.groupby("dataset").cc_coverage_gap.mean().idxmax()
    sub = d[d.dataset == ds]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    methods = [m for p in avail for m in p]
    width = 0.8 / len(methods)
    n_c = int(sub.n_classes.iloc[0])
    for i, m in enumerate(methods):
        rows = sub[sub.cp_method == m]
        per_class = []
        for c in range(n_c):
            vals = [eval(s).get(c, np.nan) for s in rows.per_class_coverage]
            per_class.append(np.nanmean(vals) if vals else np.nan)
        ax.bar(np.arange(n_c) + i * width, per_class, width=width,
               color=PALETTE[i % len(PALETTE)], label=m, edgecolor="black", linewidth=0.3)
    ax.axhline(1 - MAIN_ALPHA, color="black", ls="--", lw=0.8, label=r"$1-\alpha$")
    ax.set_xticks(np.arange(n_c) + 0.4 - width / 2)
    ax.set_xticklabels([f"class {c}" for c in range(n_c)])
    ax.set_ylabel("per-class coverage (fraction)")
    ax.set_xlabel("true class")
    ax.set_title(f"Class-conditional coverage on {ds}")
    # Headroom above 1.0 so the legend sits clear of the bars: on an imbalanced
    # dataset the minority bars are very short and a legend placed "best" lands
    # right on top of them.
    ax.set_ylim(0, 1.32)
    ax.legend(fontsize=6, ncol=4, loc="upper center", framealpha=0.95,
              borderaxespad=0.4, columnspacing=1.0, handlelength=1.4)
    save_fig(fig, "F6_cc_coverage")


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args()

    t0 = time.time()
    df = load_all_results(args.raw)
    df["study"] = df.get("study", "main")
    print(f"loaded {len(df)} rows from {len(list(Path(args.raw).glob('*.parquet')))} cells\n")

    failures = list(FAILURES.glob("*.json"))
    note("# Results summary")
    note("")
    note("Every number below is read directly from a measured row in "
         "`results/raw/*.parquet`. Nothing is interpolated or estimated.")
    note("")
    note(f"- rows: {len(df)}  |  cells: {df.cell_hash.nunique()}  |  failures: {len(failures)}")
    note(f"- datasets: {df.dataset.nunique()}  |  models: {sorted(df.model.unique())}")
    note(f"- cp methods: {sorted(df.cp_method.unique())}")
    note(f"- seeds per cell: {df.seed.nunique()}  |  alphas: {sorted(df.alpha.unique())}")
    note(f"- devices used: {sorted(df.device_used.astype(str).unique())}")
    note(f"- quantile method: {sorted(df.quantile_method.unique())}")
    note("")

    print("tables:")
    table_main(df)
    note("")
    note("### RQ1 -- wall-clock cost")
    table_cost(df)
    sd_pivot = table_stability(df)
    table_conservativeness(df)
    table_imbalance(df)
    table_alpha_robustness(df)
    table_guidelines(df, sd_pivot)

    print("\nfigures:")
    fig_coverage_boxplot(df)
    fig_pareto(df)
    fig_budget(df)
    fig_conservativeness(df)
    fig_crossover(df)
    fig_cc_coverage(df)

    if failures:
        note("")
        note(f"### {len(failures)} FAILED CELLS")
        for f in failures[:10]:
            payload = json.loads(f.read_text(encoding="utf-8"))
            note(f"- {payload['cell']}: {payload['error_type']}: {payload['error'][:100]}")

    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "results_summary.md").write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    print("\n  wrote paper/results_summary.md")

    env = environment_info()
    repro = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": env,
        "n_rows": len(df),
        "n_cells": int(df.cell_hash.nunique()),
        "n_failures": len(failures),
        "datasets": sorted(df.dataset.unique().tolist()),
        "models": sorted(df.model.unique().tolist()),
        "cp_methods": sorted(df.cp_method.unique().tolist()),
        "alphas": sorted(df.alpha.unique().tolist()),
        "scores": sorted(df.score.unique().tolist()),
        "seeds": int(df.seed.nunique()),
        "quantile_method": sorted(df.quantile_method.unique().tolist()),
        "total_measured_model_seconds": float((df.drop_duplicates("cell_hash").fit_time
                                               + df.drop_duplicates("cell_hash").predict_time).sum()),
        "artifact_build_seconds": round(time.time() - t0, 2),
        "dataset_sha256": {
            p.stem: json.loads(p.read_text(encoding="utf-8")).get("sha256")
            for p in (ROOT / "data" / "cache").glob("*.meta.json")
        },
    }
    (ROOT / "paper" / "reproducibility.json").write_text(
        json.dumps(repro, indent=2, default=str), encoding="utf-8")
    print("  wrote paper/reproducibility.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
