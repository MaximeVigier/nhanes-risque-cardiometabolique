"""Sélection des variables d'entrée du modèle.

On ne garde que ce qui répond à la question : *l'alimentation et le mode de vie,
à démographie donnée, prédisent-ils le syndrome métabolique ?*

Donc on retire explicitement (``config.LEAKAGE``) tout ce qui sert à définir la
cible (tour de taille, IMC, bilan lipidique, tension, glycémie, HbA1c, et les
traitements associés).
"""
from __future__ import annotations

import pandas as pd

from src import config

NUMERIQUES = [
    "age", "ratio_revenu",
    "energie_kcal", "proteines_g", "glucides_g", "lipides_g", "fibres_g",
    "sucres_g", "ags_g", "agmi_g", "agpi_g", "sodium_mg", "potassium_mg",
    "cholesterol_alim_mg", "cafeine_mg",
    "pct_lipides", "pct_glucides", "pct_proteines", "ratio_agpi_ags",
    "sucres_pct_kcal", "fibres_pour_1000kcal", "sodium_pour_1000kcal",
    "activite_met_min_sem", "sedentaire_min_j", "alcool_verres_sem", "sommeil_h",
]
CATEGORIELLES = ["sexe", "origine", "education", "statut_tabac", "transport_actif", "cycle"]

TOUTES = NUMERIQUES + CATEGORIELLES


def matrice_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retourne (X, y) sur les lignes où la cible est connue."""
    d = df[df["cible"].notna()].copy()
    leak = [c for c in config.LEAKAGE if c in TOUTES]
    if leak:
        raise AssertionError(f"fuite : {leak} sont à la fois features et définition de la cible")
    X = d[TOUTES].copy()
    for c in CATEGORIELLES:
        X[c] = X[c].astype("object")
    y = d["cible"].astype(int)
    return X, y
