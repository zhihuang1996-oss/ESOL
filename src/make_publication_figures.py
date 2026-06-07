"""Create publication-quality composite figures from generated ESOL results."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    """Apply a compact journal-style plotting theme."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    sns.set_style("ticks")


def save(fig: plt.Figure, name: str) -> None:
    """Save one figure as vector PDF and high-resolution PNG."""
    fig.canvas.draw()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Add bold panel label."""
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def prettify_axes(ax: plt.Axes) -> None:
    """Remove unnecessary spines and keep ticks readable."""
    sns.despine(ax=ax)
    ax.tick_params(length=3, width=0.6)


def load_esol_with_descriptors() -> pd.DataFrame:
    """Load ESOL and compute a few descriptors needed for figure panels."""
    df = pd.read_csv(ROOT / "data" / "esol.csv")
    mols = [Chem.MolFromSmiles(s.strip()) for s in df["smiles"]]
    df["MW"] = [Descriptors.MolWt(m) for m in mols]
    df["logP"] = [Descriptors.MolLogP(m) for m in mols]
    df["TPSA"] = [Descriptors.TPSA(m) for m in mols]
    df["HBD"] = [Descriptors.NumHDonors(m) for m in mols]
    df["HBA"] = [Descriptors.NumHAcceptors(m) for m in mols]
    return df


def figure_1_dataset() -> None:
    """Dataset distribution and descriptor-property associations."""
    df = load_esol_with_descriptors()
    corr = df[["MW", "logP", "TPSA", "HBD", "HBA", "logS"]].corr(numeric_only=True)["logS"].drop("logS").sort_values()
    bins = pd.read_csv(TABLES / "solubility_bin_density.csv")
    palette = ["#4059AD", "#6B9AC4", "#97D8C4", "#F4B942", "#D95D39"]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    sns.histplot(df["logS"], bins=32, kde=True, color="#2F6C8F", edgecolor="white", linewidth=0.3, ax=ax)
    ax.axvline(df["logS"].median(), color="#D1495B", lw=1.2, ls="--", label="median")
    ax.set_xlabel("Experimental logS")
    ax.set_ylabel("Molecules")
    ax.legend(frameon=False, loc="upper left")
    prettify_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    sc = ax.scatter(df["logP"], df["logS"], c=df["MW"], s=14, cmap="mako", alpha=0.78, edgecolor="none")
    ax.set_xlabel("Crippen logP")
    ax.set_ylabel("Experimental logS")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.02)
    cbar.set_label("MW")
    prettify_axes(ax)
    panel_label(ax, "b")

    ax = axes[2]
    colors = ["#D95D39" if v < 0 else "#2A9D8F" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="0.25", lw=0.7)
    ax.set_xlabel("Pearson r with logS")
    ax.set_ylabel("")
    prettify_axes(ax)
    panel_label(ax, "c")
    fig.suptitle("ESOL data curation and physicochemical context", y=1.08)
    save(fig, "Figure_1_dataset_qc")

    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.bar(bins["logS_bin"], bins["n"], color=palette, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Solubility bin")
    ax.set_ylabel("Molecules")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "Figure_S1_solubility_bins")


def figure_2_benchmark() -> None:
    """Benchmark and ablation summary."""
    bench = pd.read_csv(TABLES / "table2_benchmark_comparison.csv")
    abl = pd.read_csv(TABLES / "representation_ablation.csv")
    stats = pd.read_csv(TABLES / "statistical_tests.csv").iloc[0]
    model_labels = {
        "SVR_Descriptors": "SVR\nDesc.",
        "XGBoost_Descriptors": "XGBoost\nDesc.",
        "RandomForest_ECFP4": "RF\nECFP4",
        "MLP_ECFP4": "MLP\nECFP4",
        "KernelRidge_Fused": "KRR\nFused",
    }
    split_labels = {"random_cv": "Random CV", "scaffold_cv": "Scaffold CV"}
    rep_labels = {
        "graph_plus_fingerprints": "Graph + FP",
        "graph_proxy_only": "Graph proxy",
        "graph_fingerprint_descriptor_fusion": "Graph + FP + desc.",
        "graph_plus_descriptors": "Graph + desc.",
        "descriptors_only": "Descriptors",
        "fingerprints_only": "Fingerprints",
    }
    bench["model_label"] = bench["model"].map(model_labels).fillna(bench["model"].str.replace("_", "\n", regex=False))
    bench["split_label"] = bench["split"].map(split_labels)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.05), constrained_layout=True, gridspec_kw={"width_ratios": [1.55, 1.10]})
    ax = axes[0]
    sns.barplot(data=bench, x="model_label", y="RMSE_mean", hue="split_label", palette=["#4C78A8", "#F58518"], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("RMSE (log units)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.16), ncol=2, handlelength=1.4, columnspacing=1.1)
    ax.text(
        0.04,
        0.90,
        f"Optimism gap\n$\\Delta$RMSE = {stats['delta_RMSE']:.3f}\npaired $p$ = {stats['paired_ttest_p']:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "0.82", "lw": 0.6, "alpha": 0.92},
    )
    ax.set_ylim(0, max(bench["RMSE_mean"]) * 1.22)
    prettify_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    abl = abl.sort_values("RMSE_mean", ascending=True)
    labels = abl["representation"].map(rep_labels).fillna(abl["representation"].str.replace("_", " ", regex=False))
    y_pos = np.arange(len(abl))
    ax.errorbar(abl["RMSE_mean"], y_pos, xerr=abl["RMSE_std"], fmt="o", color="#2A9D8F", ecolor="0.62", capsize=2.5, ms=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.9", lw=0.7)
    ax.set_xlabel("Scaffold-CV RMSE")
    ax.set_ylabel("")
    prettify_axes(ax)
    panel_label(ax, "b")
    fig.suptitle("Scaffold-aware validation exposes split sensitivity and representation effects", y=1.10)
    save(fig, "Figure_2_benchmark_ablation")


def figure_3_space() -> None:
    """Chemical-space UMAP panels."""
    space = pd.read_csv(TABLES / "chemical_space_clusters.csv")
    pred = pd.read_csv(TABLES / "oof_ensemble_predictions_uncertainty.csv")
    data = space.merge(pred, on="idx", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), constrained_layout=True)
    for ax, col, title, cmap in [
        (axes[0], "logS", "Experimental logS", "viridis"),
        (axes[1], "abs_error", "OOF absolute error", "rocket_r"),
        (axes[2], "uncertainty", "OOF uncertainty", "mako_r"),
    ]:
        sc = ax.scatter(data["umap_x"], data["umap_y"], c=data[col], s=10, cmap=cmap, alpha=0.86, edgecolor="none")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.052, pad=0.02)
        prettify_axes(ax)
    for label, ax in zip(["a", "b", "c"], axes):
        panel_label(ax, label)
    fig.suptitle("Chemical-space organization of solubility, error and uncertainty", y=1.08)
    save(fig, "Figure_3_chemical_space")


def figure_4_uq_ad() -> None:
    """Uncertainty calibration and applicability-domain assessment."""
    cal = pd.read_csv(TABLES / "uncertainty_calibration.csv")
    ad = pd.read_csv(TABLES / "applicability_domain_validation.csv")
    pred = pd.read_csv(TABLES / "oof_ensemble_predictions_uncertainty.csv")

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65), constrained_layout=True)
    ax = axes[0]
    ax.plot(cal["mean_sigma"], cal["mean_abs_error"], marker="o", color="#4C78A8", lw=1.4)
    lim = max(cal["mean_sigma"].max(), cal["mean_abs_error"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="0.45", lw=1)
    ax.set_xlabel("Mean uncertainty")
    ax.set_ylabel("Mean absolute error")
    prettify_axes(ax)
    panel_label(ax, "a")

    ax = axes[1]
    tan = ad[ad["method"].str.contains("Tanimoto")].copy()
    tan["threshold"] = tan["method"].str.extract(r"([0-9]\.[0-9])").astype(float)
    ax.plot(tan["coverage"], tan["RMSE_in_AD"], marker="o", color="#2A9D8F", label="in AD")
    ax.plot(tan["coverage"], tan["RMSE_out_AD"], marker="s", color="#D95D39", label="out AD")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=False, loc="best")
    prettify_axes(ax)
    panel_label(ax, "b")

    ax = axes[2]
    ax.scatter(pred["uncertainty"], pred["abs_error"], s=9, color="#6A4C93", alpha=0.38, edgecolor="none")
    sns.regplot(data=pred, x="uncertainty", y="abs_error", scatter=False, lowess=True, color="#D1495B", ax=ax)
    ax.set_xlabel("OOF uncertainty")
    ax.set_ylabel("Absolute error")
    prettify_axes(ax)
    panel_label(ax, "c")
    fig.suptitle("Uncertainty estimates provide a reliability signal but remain imperfectly calibrated", y=1.08)
    save(fig, "Figure_4_uncertainty_ad")


def figure_5_error_sar() -> None:
    """Error and SAR diagnostics."""
    err = pd.read_csv(TABLES / "high_error_samples.csv")
    bias = pd.read_csv(TABLES / "functional_group_bias.csv")
    sar = pd.read_csv(TABLES / "sar_functional_group_importance.csv").sort_values("delta_logS")

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.20, 1.05, 1.05]})
    ax = axes[0]
    desc = ["MW", "logP", "TPSA", "RotBonds", "NumStereocenters"]
    z = err.groupby("high_error")[desc].mean()
    z = (z - err[desc].mean()) / err[desc].std(ddof=0)
    sns.heatmap(z, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "z-score"}, ax=ax)
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("High-error")
    panel_label(ax, "a")

    ax = axes[1]
    bias = bias.sort_values("mean_abs_error", ascending=True)
    ax.barh(bias["group"], bias["mean_abs_error"], color="#E76F51")
    ax.set_xlabel("Mean absolute error")
    ax.set_ylabel("")
    prettify_axes(ax)
    panel_label(ax, "b")

    ax = axes[2]
    colors = ["#D95D39" if v < 0 else "#2A9D8F" for v in sar["delta_logS"]]
    ax.barh(sar["substructure"], sar["delta_logS"], color=colors)
    ax.axvline(0, color="0.25", lw=0.7)
    ax.set_xlabel("Delta logS\npresent - absent")
    ax.set_ylabel("")
    prettify_axes(ax)
    panel_label(ax, "c")
    fig.suptitle("Error modes and chemically interpretable substructure trends", y=1.08)
    save(fig, "Figure_5_error_sar")


def graphical_abstract() -> None:
    """Create a graphical abstract for the ESOL workflow."""
    quality = pd.read_csv(TABLES / "table1_data_quality_summary.csv").set_index("metric")["value"]
    bench = pd.read_csv(TABLES / "table2_benchmark_comparison.csv")
    stats = pd.read_csv(TABLES / "statistical_tests.csv").iloc[0]
    best = bench[bench["split"] == "scaffold_cv"].sort_values("RMSE_mean").iloc[0]

    fig, ax = plt.subplots(figsize=(9.0, 3.2), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.04, 0.56, 0.17, 0.31, "ESOL data", f"{int(quality['n_valid_molecules'])} molecules\n{int(quality['n_unique_scaffolds'])} scaffolds\nlogS {quality['logS_min']:.1f} to {quality['logS_max']:.1f}"),
        (0.28, 0.56, 0.18, 0.31, "Molecular views", "RDKit descriptors\nECFP4 / MACCS\nscaffold labels"),
        (0.53, 0.56, 0.18, 0.31, "Validation", f"Random vs scaffold CV\n$\\Delta$RMSE = {stats['delta_RMSE']:.3f}\n$p$ = {stats['paired_ttest_p']:.3g}"),
        (0.78, 0.56, 0.18, 0.31, "Reliable prediction", f"Scaffold RMSE\n{best['RMSE_mean']:.3f} $\\pm$ {best['RMSE_std']:.3f}\nOOF UQ + AD"),
    ]
    colors = ["#EAF4F4", "#E8F1FA", "#FFF3D6", "#FDECE8"]
    for (x, y, w, h, title, body), color in zip(boxes, colors):
        patch = mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.018", fc=color, ec="#283845", lw=0.9)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1D3557")
        ax.text(x + w / 2, y + h - 0.17, body, ha="center", va="top", fontsize=8.0, color="#202020", linespacing=1.18)

    for x in [0.225, 0.475, 0.725]:
        ax.annotate("", xy=(x + 0.035, 0.71), xytext=(x - 0.015, 0.71), arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#495057"})

    ax.text(0.50, 0.38, "Scaffold-aware solubility modelling converts point predictions into decision-ready estimates", ha="center", fontsize=12, fontweight="bold")
    ax.plot([0.12, 0.88], [0.30, 0.30], color="#2A9D8F", lw=2.2, solid_capstyle="round")
    lower = [
        ("Split sensitivity", "random CV is optimistic"),
        ("Uncertainty", "OOF dispersion ranks risk"),
        ("Applicability domain", "coverage--RMSE trade-off"),
        ("SAR insight", "polar groups increase logS"),
    ]
    xs = [0.16, 0.39, 0.62, 0.84]
    for x, (title, body) in zip(xs, lower):
        ax.scatter([x], [0.30], s=120, color="#2A9D8F", zorder=3, edgecolor="white", linewidth=1)
        ax.text(x, 0.21, title, ha="center", fontsize=9, fontweight="bold")
        ax.text(x, 0.14, fill(body, 22), ha="center", fontsize=7.7, color="0.25")

    ax.text(0.02, 0.96, "Graphical abstract", fontsize=11, fontweight="bold", color="#1D3557")
    save(fig, "Graphical_Abstract")


def main() -> None:
    """Generate all publication figures."""
    set_style()
    figure_1_dataset()
    figure_2_benchmark()
    figure_3_space()
    figure_4_uq_ad()
    figure_5_error_sar()
    graphical_abstract()


if __name__ == "__main__":
    main()
