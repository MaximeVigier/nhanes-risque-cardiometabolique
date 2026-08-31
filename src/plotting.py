"""Figures d'exploration et de résultats (matplotlib / seaborn).

Titres en français, palette sobre, figures écrites en PNG dans
``results/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"positif": "#c1436d", "négatif": "#4f7cac"}


def save(fig: plt.Figure, name: str, *, dpi: int = 150) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def prevalence_par_groupe(df: pd.DataFrame, col: str, *, ax=None):
    ax = ax or plt.gca()
    g = (df.groupby(col)["cible"].agg(["mean", "count"])
           .sort_values("mean"))
    ax.barh(g.index.astype(str), 100 * g["mean"], color=PALETTE["négatif"])
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(100 * m + 0.5, i, f"{100*m:.0f}%  (n={n})", va="center", fontsize=9)
    ax.set_xlabel("Prévalence du syndrome métabolique (%)")
    ax.set_ylabel("")
    return ax


def distributions_nutriments(df: pd.DataFrame, cols: list[str]):
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 3 * ((n + 2) // 3)))
    for ax, c in zip(axes.ravel(), cols):
        for val, lab, color in [(1, "positif", PALETTE["positif"]), (0, "négatif", PALETTE["négatif"])]:
            s = df.loc[df["cible"] == val, c].dropna()
            q = s.quantile([0.01, 0.99])
            sns.kdeplot(s.clip(*q), ax=ax, label=lab, color=color, fill=True, alpha=0.25)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes.ravel()[0].legend(title="cible")
    fig.tight_layout()
    return fig


def courbes_pr_roc(resultats: dict[str, tuple[np.ndarray, np.ndarray]]):
    """`resultats` = {nom: (y_true, scores)}."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nom, (yt, sc) in resultats.items():
        p, r, _ = precision_recall_curve(yt, sc)
        a1.plot(r, p, label=nom)
        fpr, tpr, _ = roc_curve(yt, sc)
        a2.plot(fpr, tpr, label=nom)
    base = np.mean(next(iter(resultats.values()))[0])
    a1.axhline(base, ls="--", c="grey", lw=1, label=f"hasard ({base:.2f})")
    a1.set(xlabel="Rappel", ylabel="Précision", title="Courbe précision-rappel")
    a2.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    a2.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC")
    a1.legend(); a2.legend()
    fig.tight_layout()
    return fig


def apport_par_bloc(table: pd.DataFrame):
    """`table` indexée par nom de bloc, colonnes ROC AUC / AP."""
    fig, ax = plt.subplots(figsize=(8, 4))
    table["ROC AUC"].plot(kind="barh", ax=ax, color=PALETTE["négatif"])
    ax.set_xlabel("ROC AUC (validation croisée)")
    ax.set_xlim(0.5, max(0.85, table["ROC AUC"].max() + 0.03))
    for i, v in enumerate(table["ROC AUC"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def perf_sous_groupes(table: pd.DataFrame, metrique: str = "ROC AUC"):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(table))))
    ax.errorbar(table[metrique], range(len(table)),
                xerr=table.get("erreur"), fmt="o", color=PALETTE["positif"])
    ax.set_yticks(range(len(table)), table.index)
    ax.axvline(table[metrique].mean(), ls="--", c="grey", lw=1)
    ax.set_xlabel(f"{metrique} par sous-groupe")
    fig.tight_layout()
    return fig
