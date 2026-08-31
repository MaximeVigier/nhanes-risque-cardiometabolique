"""Construction de la cible : syndrome métabolique (NCEP ATP III révisé).

Cinq critères, chacun vrai si la mesure dépasse le seuil OU si la personne est
traitée pour ce facteur (définition harmonisée Alberti 2009) :

1. tour de taille élevé (seuil selon le sexe) ;
2. triglycérides ≥ 150 mg/dL ;
3. HDL bas (< 40 homme, < 50 femme) ;
4. pression ≥ 130/85 mmHg, ou sous antihypertenseur ;
5. glycémie à jeun ≥ 100 mg/dL, ou sous antidiabétique.

Syndrome métabolique = au moins 3 critères sur 5.

Le HDL bas et la tension traitée comptent comme critères même si la valeur
mesurée est « normale » : c'est le facteur de risque qui compte, pas seulement
la mesure du jour. En revanche les diabétiques et les personnes avec MCV connue
sont sortis de l'échantillon en amont (``preprocessing.filtre_population``) :
on modélise un risque, pas une maladie déjà là.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def criteres(df: pd.DataFrame) -> pd.DataFrame:
    n = config.NCEP
    seuil_taille = df["sexe"].map(n["tour_taille_cm"])
    seuil_hdl = df["sexe"].map(n["hdl"])

    c = pd.DataFrame(index=df.index)
    c["taille"] = df["tour_taille_cm"] > seuil_taille
    c["trigly"] = df["triglycerides"] >= n["triglycerides"]
    c["hdl"] = df["hdl"] < seuil_hdl
    c["tension"] = (df["pas"] >= n["pas"]) | (df["pad"] >= n["pad"]) | df["traite_hta"].fillna(False)
    c["glycemie"] = (df["glycemie_jeun"] >= n["glycemie_jeun"]) | df["traite_diabete"].fillna(False)

    # on garde le NaN quand la mesure manque (sauf si déjà rendu vrai par le traitement)
    c["taille"] = c["taille"].where(df["tour_taille_cm"].notna())
    c["trigly"] = c["trigly"].where(df["triglycerides"].notna())
    c["hdl"] = c["hdl"].where(df["hdl"].notna())
    c["tension"] = c["tension"].where(df["pas"].notna() | df["traite_hta"].fillna(False))
    c["glycemie"] = c["glycemie"].where(df["glycemie_jeun"].notna() | df["traite_diabete"].fillna(False))
    return c


def syndrome_metabolique(df: pd.DataFrame, min_criteres: int = 3) -> pd.Series:
    """Série 0/1 ; NaN si le nombre de critères manquants empêche de trancher
    (moins de 3 confirmés et pas assez de critères mesurés pour exclure)."""
    c = criteres(df)
    n_vrais = c.eq(True).sum(axis=1)
    n_inconnus = c.isna().sum(axis=1)

    positif = n_vrais >= min_criteres
    negatif_certain = (n_vrais + n_inconnus) < min_criteres  # 3 hors d'atteinte
    y = pd.Series(np.nan, index=df.index)
    y[positif] = 1.0
    y[~positif & negatif_certain] = 0.0
    return y
