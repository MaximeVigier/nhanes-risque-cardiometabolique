"""Tests de la construction de la cible « syndrome métabolique »."""
import numpy as np
import pandas as pd

from src import target


def _base(**kw):
    """Un participant « tout normal », modifiable par mot-clé."""
    row = dict(sexe="Homme", tour_taille_cm=90.0, triglycerides=100.0, hdl=55.0,
               pas=115.0, pad=70.0, glycemie_jeun=90.0, traite_hta=False,
               traite_diabete=False)
    row.update(kw)
    return pd.DataFrame([row])


def test_aucun_critere():
    assert target.syndrome_metabolique(_base()).iloc[0] == 0.0


def test_deux_criteres_reste_negatif():
    df = _base(tour_taille_cm=110.0, triglycerides=180.0)
    assert target.syndrome_metabolique(df).iloc[0] == 0.0


def test_trois_criteres_positif():
    df = _base(tour_taille_cm=110.0, triglycerides=180.0, glycemie_jeun=105.0)
    assert target.syndrome_metabolique(df).iloc[0] == 1.0


def test_seuils_dependent_du_sexe():
    # tour de taille 95 cm : sous le seuil homme (102), au-dessus du seuil femme (88)
    assert not target.criteres(_base(sexe="Homme", tour_taille_cm=95.0))["taille"].iloc[0]
    assert target.criteres(_base(sexe="Femme", tour_taille_cm=95.0))["taille"].iloc[0]


def test_traitement_hta_compte_comme_critere():
    df = _base(pas=118.0, pad=72.0, traite_hta=True)
    assert target.criteres(df)["tension"].iloc[0]


def test_indetermine_si_trop_de_mesures_manquantes():
    df = _base(tour_taille_cm=110.0, triglycerides=np.nan, hdl=np.nan,
               pas=np.nan, glycemie_jeun=np.nan)
    # 1 critère vrai, 4 inconnus -> on ne peut ni confirmer (>=3) ni exclure
    assert pd.isna(target.syndrome_metabolique(df).iloc[0])
