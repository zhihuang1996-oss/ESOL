"""Reproducible ESOL scaffold-aware solubility analysis.

The script intentionally keeps the default run lightweight enough for a local
workstation while producing data-grounded tables and figures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
RDLogger.DisableLog("rdApp.warning")
sns.set_theme(style="whitegrid", context="paper")


BASIC_DESCRIPTOR_FUNCS = {
    "MW": Descriptors.MolWt,
    "logP": Descriptors.MolLogP,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "RotBonds": Descriptors.NumRotatableBonds,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumStereocenters": lambda mol: len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    "MolRefractivity": Descriptors.MolMR,
}

FUNCTIONAL_GROUPS = {
    "hydroxyl": "[OX2H]",
    "carboxylic_acid": "C(=O)[OX2H1]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "C(=O)N",
    "halogen": "[F,Cl,Br,I]",
    "aromatic_ring": "a1aaaaa1",
    "ether": "[OD2]([#6])[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "nitrile": "C#N",
    "sulfonamide": "S(=O)(=O)N",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "long_aliphatic": "[CX4][CX4][CX4][CX4][CX4]",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set all available random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create output directories."""
    for key in ["results_dir", "figures_dir", "tables_dir", "models_dir"]:
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def canonicalize_smiles(smiles: str) -> tuple[str | None, Chem.Mol | None]:
    """Convert a SMILES string to RDKit canonical SMILES and molecule."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    parent = Chem.Mol(mol)
    Chem.SanitizeMol(parent)
    return Chem.MolToSmiles(parent, canonical=True), parent


def load_esol(config: dict[str, Any]) -> pd.DataFrame:
    """Load ESOL, canonicalize SMILES, and drop invalid molecules."""
    df = pd.read_csv(config["data_path"])
    records = []
    for idx, row in df.iterrows():
        can, mol = canonicalize_smiles(row[config["smiles_col"]])
        if mol is not None:
            records.append(
                {
                    "idx": idx,
                    "smiles": row[config["smiles_col"]],
                    "canonical_smiles": can,
                    "logS": float(row[config["target"]]),
                    "mol": mol,
                }
            )
    return pd.DataFrame(records)


def compute_basic_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    """Compute the 12 requested RDKit molecular descriptors."""
    rows = []
    for mol in mols:
        rows.append({name: func(mol) for name, func in BASIC_DESCRIPTOR_FUNCS.items()})
    return pd.DataFrame(rows)


def compute_all_descriptors(mols: list[Chem.Mol]) -> pd.DataFrame:
    """Compute numeric RDKit descriptors and replace invalid values."""
    rows = []
    for mol in mols:
        values = {}
        for name, func in Descriptors._descList:
            try:
                values[name] = float(func(mol))
            except Exception:
                values[name] = np.nan
        rows.append(values)
    desc = pd.DataFrame(rows)
    desc = desc.replace([np.inf, -np.inf], np.nan)
    return desc.fillna(desc.median(numeric_only=True))


def bitvect_to_array(bitvect: Any, n_bits: int) -> np.ndarray:
    """Convert an RDKit bit vector to a numpy array."""
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def compute_fingerprints(mols: list[Chem.Mol], config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute ECFP4, MACCS, and RDKit topological fingerprints."""
    ecfp_bits = int(config["features"]["ecfp_bits"])
    rdkit_bits = int(config["features"]["rdkit_fp_bits"])
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=rdkit_bits)
    ecfp, maccs, rdk = [], [], []
    for mol in mols:
        ecfp.append(bitvect_to_array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ecfp_bits), ecfp_bits))
        maccs.append(bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), 167))
        rdk.append(bitvect_to_array(rdkit_gen.GetFingerprint(mol), rdkit_bits))
    return {"ECFP4": np.asarray(ecfp), "MACCS": np.asarray(maccs), "RDKFingerprint": np.asarray(rdk)}


def compute_scaffolds(mols: list[Chem.Mol]) -> list[str]:
    """Compute Bemis-Murcko scaffold SMILES with acyclic fallback labels."""
    scaffolds = []
    for mol in mols:
        smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if not smi:
            smi = "acyclic_" + Chem.MolToSmiles(mol, canonical=True)[:24]
        scaffolds.append(smi)
    return scaffolds


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return regression metrics."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error with broad scikit-learn compatibility."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_model(name: str, config: dict[str, Any], seed: int) -> Any:
    """Construct a regression model by name."""
    if name == "RandomForest_ECFP4":
        params = config["models"]["random_forest"]
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "XGBoost_Descriptors":
        if XGBRegressor is None:
            return RandomForestRegressor(random_state=seed, n_jobs=-1, n_estimators=200)
        params = config["models"]["xgboost"]
        return XGBRegressor(random_state=seed, objective="reg:squarederror", n_jobs=1, **params)
    if name == "SVR_Descriptors":
        params = config["models"]["svr"]
        return make_pipeline(StandardScaler(), SVR(**params))
    if name == "MLP_ECFP4":
        params = config["models"]["mlp"]
        return make_pipeline(StandardScaler(with_mean=False), MLPRegressor(random_state=seed, early_stopping=True, **params))
    if name == "KernelRidge_Fused":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01))
    raise ValueError(f"Unknown model: {name}")


def scaffold_kfold(scaffolds: list[str], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return scaffold-grouped cross-validation splits."""
    groups = np.asarray(scaffolds)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def random_kfold(n_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return random cross-validation splits."""
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n_samples)))


def evaluate_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
    split_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate one model under supplied splits and return fold metrics and OOF predictions."""
    rows, oof = [], np.full_like(y, np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = make_model(model_name, config, seed + fold)
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float)
        oof[test_idx] = pred
        row = {"model": model_name, "split": split_name, "fold": fold, **metrics(y[test_idx], pred)}
        rows.append(row)
    return pd.DataFrame(rows), oof


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold metrics as mean and standard deviation."""
    summary = []
    for (model, split), group in cv_df.groupby(["model", "split"]):
        row = {"model": model, "split": split}
        for metric in ["RMSE", "MAE", "R2"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary.append(row)
    return pd.DataFrame(summary).sort_values(["split", "RMSE_mean"])


def fixed_scaffold_split(scaffolds: list[str], y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic scaffold train/validation/test split."""
    scaffold_df = pd.DataFrame({"scaffold": scaffolds, "y": y})
    grouped = scaffold_df.groupby("scaffold").agg(n=("y", "size"), mean_y=("y", "mean")).reset_index()
    grouped = grouped.sample(frac=1.0, random_state=seed).sort_values("n", ascending=False)
    total = len(y)
    target_test = 0.20 * total
    target_val = 0.10 * total
    bins = {"test": [], "val": [], "train": []}
    counts = {"test": 0, "val": 0, "train": 0}
    for _, row in grouped.iterrows():
        if counts["test"] < target_test:
            dest = "test"
        elif counts["val"] < target_val:
            dest = "val"
        else:
            dest = "train"
        bins[dest].append(row["scaffold"])
        counts[dest] += int(row["n"])
    labels = np.asarray(scaffolds)
    return (
        np.where(np.isin(labels, bins["train"]))[0],
        np.where(np.isin(labels, bins["val"]))[0],
        np.where(np.isin(labels, bins["test"]))[0],
    )


def tanimoto_matrix(test_fps: list[Any], train_fps: list[Any]) -> np.ndarray:
    """Compute max Tanimoto similarity from each test molecule to training set."""
    return np.asarray([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in test_fps])


def plot_histogram(df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot logS distribution."""
    plt.figure(figsize=(6, 4))
    sns.histplot(df["logS"], bins=35, kde=True, color="#3b7ea1")
    plt.xlabel("Experimental logS (mol/L)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "logs_distribution.png", dpi=dpi)
    plt.close()


def plot_correlations(data: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot Pearson and Spearman descriptor correlations with logS."""
    for method in ["pearson", "spearman"]:
        corr = data.corr(method=method, numeric_only=True)[["logS"]].drop("logS").sort_values("logS")
        plt.figure(figsize=(4.8, 6.2))
        sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": method})
        plt.title(f"{method.title()} correlation with logS")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{method}_descriptor_correlation.png", dpi=dpi)
        plt.close()


def plot_embedding(embedding: np.ndarray, color: Any, title: str, path: Path, dpi: int, categorical: bool = False) -> None:
    """Plot a 2D chemical-space embedding."""
    plt.figure(figsize=(6, 5))
    if categorical:
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=color, s=18, linewidth=0, palette="tab10")
        plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=color, s=18, cmap="viridis", linewidths=0)
        plt.colorbar(sc, label=title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_ad_curve(ad_curve: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot AD coverage versus RMSE."""
    plt.figure(figsize=(5.5, 4))
    sns.lineplot(data=ad_curve, x="coverage", y="RMSE_in_AD", hue="method", marker="o")
    plt.xlabel("AD coverage")
    plt.ylabel("RMSE within AD")
    plt.tight_layout()
    plt.savefig(figures_dir / "ad_coverage_rmse_curve.png", dpi=dpi)
    plt.close()


def calibration_table(y_true: np.ndarray, pred: np.ndarray, sigma: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Compute uncertainty calibration bins."""
    err = np.abs(y_true - pred)
    bins = pd.qcut(sigma, q=n_bins, duplicates="drop")
    rows = []
    for interval in bins.categories:
        mask = bins == interval
        rows.append(
            {
                "uncertainty_bin": str(interval),
                "mean_sigma": float(np.mean(sigma[mask])),
                "mean_abs_error": float(np.mean(err[mask])),
                "coverage_90": float(np.mean(err[mask] <= 1.64 * sigma[mask])),
                "n": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(calib: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    """Plot calibration curve."""
    plt.figure(figsize=(5, 4))
    plt.plot(calib["mean_sigma"], calib["mean_abs_error"], marker="o", label="Observed")
    max_val = max(calib["mean_sigma"].max(), calib["mean_abs_error"].max())
    plt.plot([0, max_val], [0, max_val], "--", color="gray", label="Ideal")
    plt.xlabel("Mean predictive uncertainty")
    plt.ylabel("Mean absolute error")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / "uncertainty_calibration.png", dpi=dpi)
    plt.close()


def plot_error_boxes(df: pd.DataFrame, descriptors: list[str], figures_dir: Path, dpi: int) -> None:
    """Plot descriptor distributions for high-error samples."""
    long_df = df.melt(id_vars=["high_error"], value_vars=descriptors, var_name="descriptor", value_name="value")
    g = sns.catplot(data=long_df, x="high_error", y="value", col="descriptor", kind="box", col_wrap=3, sharey=False)
    g.set_axis_labels("High-error sample", "Descriptor value")
    g.fig.tight_layout()
    g.fig.savefig(figures_dir / "high_error_descriptor_boxes.png", dpi=dpi)
    plt.close(g.fig)


def draw_molecule_panel(df: pd.DataFrame, path: Path, legends: list[str]) -> None:
    """Draw a grid of representative molecules."""
    mols = df["mol"].tolist()
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(240, 180), legends=legends, useSVG=False)
    img.save(str(path))


def run_pipeline(config: dict[str, Any]) -> None:
    """Run the complete ESOL project workflow."""
    set_seed(int(config["seed"]))
    ensure_dirs(config)
    figures_dir = Path(config["figures_dir"])
    tables_dir = Path(config["tables_dir"])
    dpi = int(config["plots"]["dpi"])

    df = load_esol(config)
    mols = df["mol"].tolist()
    y = df["logS"].to_numpy()
    basic = compute_basic_descriptors(mols)
    all_desc = compute_all_descriptors(mols)
    fps = compute_fingerprints(mols, config)
    scaffolds = compute_scaffolds(mols)
    df["scaffold"] = scaffolds
    df["fold"] = -1

    for fold, (_, test_idx) in enumerate(scaffold_kfold(scaffolds, int(config["cv"]["n_splits"]))):
        df.loc[test_idx, "fold"] = fold

    missing_rate = basic.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "descriptor"})
    quality = pd.DataFrame(
        [
            {"metric": "n_raw", "value": len(pd.read_csv(config["data_path"]))},
            {"metric": "n_valid_molecules", "value": len(df)},
            {"metric": "n_unique_raw_smiles", "value": df["smiles"].nunique()},
            {"metric": "n_unique_canonical_smiles", "value": df["canonical_smiles"].nunique()},
            {"metric": "duplicate_canonical_smiles", "value": len(df) - df["canonical_smiles"].nunique()},
            {"metric": "n_unique_scaffolds", "value": pd.Series(scaffolds).nunique()},
            {"metric": "logS_mean", "value": float(np.mean(y))},
            {"metric": "logS_std", "value": float(np.std(y, ddof=1))},
            {"metric": "logS_min", "value": float(np.min(y))},
            {"metric": "logS_max", "value": float(np.max(y))},
        ]
    )
    sol_bins = pd.cut(y, bins=[-np.inf, -6, -4, -2, 0, np.inf], labels=["very_low", "low", "moderate", "high", "very_high"])
    sol_density = sol_bins.value_counts().rename_axis("logS_bin").reset_index(name="n")
    quality.to_csv(tables_dir / "table1_data_quality_summary.csv", index=False)
    missing_rate.to_csv(tables_dir / "descriptor_missing_rates.csv", index=False)
    sol_density.to_csv(tables_dir / "solubility_bin_density.csv", index=False)

    data_for_corr = pd.concat([basic, pd.Series(y, name="logS")], axis=1)
    plot_histogram(df, figures_dir, dpi)
    plot_correlations(data_for_corr, figures_dir, dpi)

    pca = PCA(n_components=int(config["features"]["descriptor_pca_components"]), random_state=int(config["seed"]))
    desc_pca = pca.fit_transform(StandardScaler().fit_transform(all_desc))
    X_desc = np.asarray(all_desc, dtype=float)
    X_desc_pca = np.asarray(desc_pca, dtype=float)
    X_ecfp = fps["ECFP4"].astype(float)
    X_maccs = fps["MACCS"].astype(float)
    X_graph_proxy = np.asarray(basic, dtype=float)
    X_fp = np.hstack([X_ecfp, X_maccs])
    X_fused = np.hstack([X_ecfp, X_maccs, X_desc_pca, StandardScaler().fit_transform(X_graph_proxy)])

    random_splits = random_kfold(len(df), int(config["cv"]["n_splits"]), int(config["seed"]))
    scaffold_splits = scaffold_kfold(scaffolds, int(config["cv"]["n_splits"]))
    model_features = {
        "RandomForest_ECFP4": X_ecfp,
        "XGBoost_Descriptors": X_desc,
        "SVR_Descriptors": X_desc,
        "MLP_ECFP4": X_ecfp,
        "KernelRidge_Fused": X_fused,
    }
    cv_frames, oof_preds = [], {}
    for split_name, splits in [("random_cv", random_splits), ("scaffold_cv", scaffold_splits)]:
        for model_name, X in model_features.items():
            fold_df, oof = evaluate_cv(model_name, X, y, splits, config, int(config["seed"]), split_name)
            cv_frames.append(fold_df)
            oof_preds[(split_name, model_name)] = oof
    cv_df = pd.concat(cv_frames, ignore_index=True)
    cv_df.to_csv(tables_dir / "cv_fold_metrics.csv", index=False)
    cv_summary = summarize_cv(cv_df)
    cv_summary.to_csv(tables_dir / "table2_benchmark_comparison.csv", index=False)

    random_best = cv_summary[cv_summary["split"] == "random_cv"].sort_values("RMSE_mean").iloc[0]
    scaffold_best = cv_summary[cv_summary["split"] == "scaffold_cv"].sort_values("RMSE_mean").iloc[0]
    paired = cv_df.pivot_table(index="fold", columns="split", values="RMSE", aggfunc="min")
    optimism_p = stats.ttest_rel(paired["scaffold_cv"], paired["random_cv"]).pvalue if set(paired.columns) >= {"random_cv", "scaffold_cv"} else np.nan
    stats_table = pd.DataFrame(
        [
            {
                "comparison": "best_random_cv_vs_best_scaffold_cv_RMSE",
                "delta_RMSE": float(scaffold_best["RMSE_mean"] - random_best["RMSE_mean"]),
                "paired_ttest_p": float(optimism_p),
            }
        ]
    )
    stats_table.to_csv(tables_dir / "statistical_tests.csv", index=False)

    ablation_features = {
        "graph_proxy_only": X_graph_proxy,
        "fingerprints_only": X_fp,
        "descriptors_only": X_desc_pca,
        "graph_plus_fingerprints": np.hstack([StandardScaler().fit_transform(X_graph_proxy), X_fp]),
        "graph_plus_descriptors": np.hstack([StandardScaler().fit_transform(X_graph_proxy), X_desc_pca]),
        "graph_fingerprint_descriptor_fusion": X_fused,
    }
    ablation_rows = []
    for name, X in ablation_features.items():
        folds, _ = evaluate_cv("RandomForest_ECFP4", X, y, scaffold_splits, config, int(config["seed"]), "scaffold_cv")
        row = {"representation": name}
        for metric_name in ["RMSE", "MAE", "R2"]:
            row[f"{metric_name}_mean"] = folds[metric_name].mean()
            row[f"{metric_name}_std"] = folds[metric_name].std(ddof=1)
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).sort_values("RMSE_mean").to_csv(tables_dir / "representation_ablation.csv", index=False)

    ensemble_members = ["RandomForest_ECFP4", "XGBoost_Descriptors", "SVR_Descriptors", "MLP_ECFP4", "KernelRidge_Fused"]
    member_oofs = np.vstack([oof_preds[("scaffold_cv", m)] for m in ensemble_members])
    ens_pred = np.nanmean(member_oofs, axis=0)
    ens_sigma = np.nanstd(member_oofs, axis=0, ddof=1)
    ens_sigma = np.maximum(ens_sigma, np.percentile(ens_sigma, 10))
    uq_df = pd.DataFrame({"idx": df["idx"], "logS": y, "prediction": ens_pred, "abs_error": np.abs(y - ens_pred), "uncertainty": ens_sigma})
    uq_df.to_csv(tables_dir / "oof_ensemble_predictions_uncertainty.csv", index=False)
    calib = calibration_table(y, ens_pred, ens_sigma)
    calib.to_csv(tables_dir / "uncertainty_calibration.csv", index=False)
    plot_calibration(calib, figures_dir, dpi)

    if umap is not None:
        if bool(config["plots"].get("use_fast_umap", True)):
            umap_input = PCA(n_components=32, random_state=int(config["seed"])).fit_transform(X_ecfp)
            reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean", n_epochs=120, random_state=int(config["seed"]))
            emb = reducer.fit_transform(umap_input)
        else:
            reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="jaccard", random_state=int(config["seed"]))
            emb = reducer.fit_transform(X_ecfp.astype(bool))
    else:
        emb = PCA(n_components=2, random_state=int(config["seed"])).fit_transform(X_ecfp)
    plot_embedding(emb, y, "Experimental logS", figures_dir / "umap_logs.png", dpi)
    plot_embedding(emb, df["fold"].astype(str), "Scaffold CV fold", figures_dir / "umap_scaffold_fold.png", dpi, categorical=True)
    plot_embedding(emb, np.abs(y - ens_pred), "Absolute prediction error", figures_dir / "umap_prediction_error.png", dpi)
    tsne_emb = TSNE(n_components=2, random_state=int(config["seed"]), init="pca", learning_rate="auto", perplexity=30).fit_transform(X_ecfp)
    plot_embedding(tsne_emb, y, "t-SNE logS", figures_dir / "tsne_logs.png", dpi)
    clusters = DBSCAN(eps=0.75, min_samples=8).fit_predict(emb)
    pd.DataFrame({"idx": df["idx"], "umap_x": emb[:, 0], "umap_y": emb[:, 1], "cluster": clusters, "scaffold": scaffolds}).to_csv(
        tables_dir / "chemical_space_clusters.csv", index=False
    )

    train_idx, val_idx, test_idx = fixed_scaffold_split(scaffolds, y, int(config["seed"]))
    test_y = y[test_idx]
    ensemble_test_preds = []
    for model_name, X in model_features.items():
        model = make_model(model_name, config, int(config["seed"]))
        fit_idx = np.concatenate([train_idx, val_idx])
        model.fit(X[fit_idx], y[fit_idx])
        ensemble_test_preds.append(np.asarray(model.predict(X[test_idx]), dtype=float))
    test_pred = np.mean(ensemble_test_preds, axis=0)
    test_sigma = np.maximum(np.std(ensemble_test_preds, axis=0, ddof=1), 1e-3)

    rdkit_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=int(config["features"]["ecfp_bits"])) for m in mols]
    sim = tanimoto_matrix([rdkit_fps[i] for i in test_idx], [rdkit_fps[i] for i in train_idx])
    ad_rows, ad_curve_rows = [], []
    for threshold in config["ad"]["tanimoto_thresholds"]:
        mask = sim >= float(threshold)
        if np.any(mask) and np.any(~mask):
            rmse_in = rmse(test_y[mask], test_pred[mask])
            rmse_out = rmse(test_y[~mask], test_pred[~mask])
            pval = stats.ttest_ind(np.abs(test_y[mask] - test_pred[mask]), np.abs(test_y[~mask] - test_pred[~mask]), equal_var=False).pvalue
        else:
            rmse_in, rmse_out, pval = np.nan, np.nan, np.nan
        ad_rows.append({"method": f"Tanimoto >= {threshold}", "coverage": float(np.mean(mask)), "RMSE_in_AD": rmse_in, "RMSE_out_AD": rmse_out, "p_value": pval})
        ad_curve_rows.append({"method": "Tanimoto", "coverage": float(np.mean(mask)), "RMSE_in_AD": rmse_in})

    desc_scaler = StandardScaler().fit(X_desc_pca[train_idx])
    x_train_desc = desc_scaler.transform(X_desc_pca[train_idx])
    x_test_desc = desc_scaler.transform(X_desc_pca[test_idx])
    xtx_inv = np.linalg.pinv(x_train_desc.T @ x_train_desc)
    leverage = np.einsum("ij,jk,ik->i", x_test_desc, xtx_inv, x_test_desc)
    lev_threshold = 3 * x_train_desc.shape[1] / x_train_desc.shape[0]
    lev_mask = leverage <= lev_threshold
    ad_rows.append(
        {
            "method": "Leverage",
            "coverage": float(np.mean(lev_mask)),
            "RMSE_in_AD": rmse(test_y[lev_mask], test_pred[lev_mask]) if np.any(lev_mask) else np.nan,
            "RMSE_out_AD": rmse(test_y[~lev_mask], test_pred[~lev_mask]) if np.any(~lev_mask) else np.nan,
            "p_value": np.nan,
        }
    )
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(x_train_desc)
    density = kde.score_samples(x_test_desc)
    density_cut = np.quantile(kde.score_samples(x_train_desc), float(config["ad"]["kde_quantile"]))
    kde_mask = density >= density_cut
    ad_rows.append(
        {
            "method": "KDE density",
            "coverage": float(np.mean(kde_mask)),
            "RMSE_in_AD": rmse(test_y[kde_mask], test_pred[kde_mask]) if np.any(kde_mask) else np.nan,
            "RMSE_out_AD": rmse(test_y[~kde_mask], test_pred[~kde_mask]) if np.any(~kde_mask) else np.nan,
            "p_value": np.nan,
        }
    )
    ad_table = pd.DataFrame(ad_rows)
    ad_table.to_csv(tables_dir / "applicability_domain_validation.csv", index=False)
    plot_ad_curve(pd.DataFrame(ad_curve_rows), figures_dir, dpi)

    test_frame = df.iloc[test_idx].copy()
    test_frame["prediction"] = test_pred
    test_frame["abs_error"] = np.abs(test_y - test_pred)
    test_frame["uncertainty"] = test_sigma
    test_frame["in_AD_tanimoto_0.4"] = sim >= 0.4
    threshold = 1.5 * mean_absolute_error(test_y, test_pred)
    test_frame["high_error"] = test_frame["abs_error"] > threshold
    error_df = pd.concat([test_frame.reset_index(drop=True), basic.iloc[test_idx].reset_index(drop=True)], axis=1)
    error_df.drop(columns=["mol"]).to_csv(tables_dir / "high_error_samples.csv", index=False)
    plot_error_boxes(error_df, ["MW", "logP", "TPSA", "RotBonds", "NumStereocenters"], figures_dir, dpi)

    bias_rows = []
    for group, smarts in FUNCTIONAL_GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        mask = np.asarray([mol.HasSubstructMatch(patt) for mol in test_frame["mol"]])
        if np.any(mask):
            residual = test_frame.loc[mask, "prediction"].to_numpy() - test_frame.loc[mask, "logS"].to_numpy()
            bias_rows.append({"group": group, "n": int(mask.sum()), "mean_residual": float(np.mean(residual)), "mean_abs_error": float(np.mean(np.abs(residual)))})
    bias_table = pd.DataFrame(bias_rows).sort_values("mean_abs_error", ascending=False)
    bias_table.to_csv(tables_dir / "functional_group_bias.csv", index=False)
    plt.figure(figsize=(6, 4.5))
    bias_pivot = bias_table.set_index("group")[["mean_residual", "mean_abs_error"]]
    sns.heatmap(bias_pivot, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.tight_layout()
    plt.savefig(figures_dir / "systematic_bias_heatmap.png", dpi=dpi)
    plt.close()

    fg_rows = []
    for group, smarts in FUNCTIONAL_GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        presence = np.asarray([mol.HasSubstructMatch(patt) for mol in mols])
        if presence.sum() >= 5:
            fg_rows.append(
                {
                    "substructure": group,
                    "n": int(presence.sum()),
                    "mean_logS_present": float(np.mean(y[presence])),
                    "mean_logS_absent": float(np.mean(y[~presence])),
                    "delta_logS": float(np.mean(y[presence]) - np.mean(y[~presence])),
                    "mannwhitney_p": float(stats.mannwhitneyu(y[presence], y[~presence], alternative="two-sided").pvalue),
                }
            )
    fg_table = pd.DataFrame(fg_rows).sort_values("delta_logS", ascending=False)
    fg_table.to_csv(tables_dir / "sar_functional_group_importance.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=fg_table, y="substructure", x="delta_logS", palette="vlag")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean logS difference: present - absent")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "functional_group_importance_top20.png", dpi=dpi)
    plt.close()

    representatives = pd.concat(
        [
            df.nsmallest(3, "logS"),
            df.nlargest(3, "logS"),
            test_frame.nlargest(4, "abs_error"),
        ],
        ignore_index=True,
    ).drop_duplicates("canonical_smiles").head(10)
    legends = [f"logS={row.logS:.2f}" for row in representatives.itertuples()]
    draw_molecule_panel(representatives, figures_dir / "representative_molecules.png", legends)

    screening = test_frame.copy()
    screening["screening_class"] = np.select(
        [
            (screening["in_AD_tanimoto_0.4"]) & (screening["prediction"] > -2),
            (screening["in_AD_tanimoto_0.4"]) & (screening["prediction"] < -5),
            ~screening["in_AD_tanimoto_0.4"],
        ],
        ["high_confidence_high_solubility", "high_confidence_low_solubility_warning", "outside_AD_unreliable"],
        default="moderate",
    )
    screening[["canonical_smiles", "logS", "prediction", "uncertainty", "in_AD_tanimoto_0.4", "screening_class"]].sort_values(
        ["in_AD_tanimoto_0.4", "prediction"], ascending=[False, False]
    ).to_csv(tables_dir / "virtual_screening_local_demo.csv", index=False)
    draw_molecule_panel(screening.sort_values("uncertainty").head(5), figures_dir / "screening_representative_molecules.png", [f"pred={v:.2f}" for v in screening.sort_values("uncertainty").head(5)["prediction"]])

    artifact_index = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": len(df),
        "figures": sorted(p.name for p in figures_dir.glob("*.png")),
        "tables": sorted(p.name for p in tables_dir.glob("*.csv")),
    }
    with open(Path(config["results_dir"]) / "artifact_index.json", "w", encoding="utf-8") as handle:
        json.dump(artifact_index, handle, indent=2)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(load_config(args.config))
