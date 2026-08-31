"""Tests des utilitaires de nettoyage."""
import numpy as np
import pandas as pd

from src import config, preprocessing
from src.features import matrice_X_y


def test_fix_xpt_zeros():
    s = pd.Series([5.4e-79, 0.0, 3.0, -2e-80])
    out = preprocessing._fix_xpt_zeros(s)
    assert out.tolist() == [0.0, 0.0, 3.0, 0.0]


def test_blank_special_met_les_codes_en_nan():
    df = pd.DataFrame({"ALQ130": [2.0, 777.0, 999.0, 5.0]})
    out = preprocessing._blank_special(df)
    assert out["ALQ130"].isna().tolist() == [False, True, True, False]


def test_pas_de_fuite_dans_les_features():
    """Aucune variable de définition de la cible ne doit être une feature."""
    communs = set(config.LEAKAGE) & set(
        __import__("src.features", fromlist=["TOUTES"]).TOUTES
    )
    assert communs == set()


def test_matrice_X_y_rejette_les_cibles_manquantes():
    df = pd.DataFrame({
        "cible": [1.0, np.nan, 0.0],
        **{c: [1, 2, 3] for c in __import__("src.features", fromlist=["NUMERIQUES"]).NUMERIQUES},
        **{c: ["a", "b", "c"] for c in __import__("src.features", fromlist=["CATEGORIELLES"]).CATEGORIELLES},
    })
    X, y = matrice_X_y(df)
    assert len(X) == len(y) == 2
