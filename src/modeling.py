"""Pipelines de modélisation, validation croisée, évaluation.

Choix :

* pré-traitement dans le pipeline (imputation, standardisation, one-hot) pour ne
  rien apprendre du jeu de test pendant la CV ;
* métrique de suivi principale : average precision (aire sous la courbe
  précision-rappel), adaptée au déséquilibre (~1/3 de positifs). On regarde
  aussi ROC AUC, le score de Brier (calibration) et le rappel à précision fixée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import features

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def preprocesseur(pour_arbres: bool) -> ColumnTransformer:
    num = features.NUMERIQUES
    cat = features.CATEGORIELLES
    if pour_arbres:
        num_pipe = SimpleImputer(strategy="median", add_indicator=True)
    else:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def modeles() -> dict[str, Pipeline]:
    return {
        "Régression logistique": Pipeline([
            ("prep", preprocesseur(pour_arbres=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random forest": Pipeline([
            ("prep", preprocesseur(pour_arbres=True)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                           class_weight="balanced_subsample",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", preprocesseur(pour_arbres=True)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                                   max_iter=600, l2_regularization=1.0,
                                                   random_state=RANDOM_STATE)),
        ]),
    }


def comparer(models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    lignes = []
    for nom, pipe in models.items():
        r = cross_validate(pipe, X, y, cv=CV, n_jobs=-1,
                           scoring=["average_precision", "roc_auc", "neg_brier_score"])
        lignes.append({
            "modèle": nom,
            "AP (PR-AUC)": r["test_average_precision"].mean(),
            "AP ± σ": r["test_average_precision"].std(),
            "ROC AUC": r["test_roc_auc"].mean(),
            "Brier": -r["test_neg_brier_score"].mean(),
        })
    return pd.DataFrame(lignes).set_index("modèle").round(4)


def ciblage_top(y_true, scores, part: float = 0.2) -> dict:
    """Si on cible la fraction `part` jugée la plus à risque : quelle part des cas
    capture-t-on (rappel), et quelle est la précision dans cette fraction ?"""
    y_true = np.asarray(y_true)
    k = max(1, int(round(part * len(scores))))
    top = np.argsort(scores)[::-1][:k]
    capte = y_true[top].sum()
    return {
        "precision": capte / k,
        "rappel": capte / y_true.sum(),
        "lift": (capte / k) / y_true.mean(),
    }


def evaluer_test(pipe, X_tr, y_tr, X_te, y_te) -> dict:
    pipe.fit(X_tr, y_tr)
    s = pipe.predict_proba(X_te)[:, 1]
    out = {
        "AP (PR-AUC)": average_precision_score(y_te, s),
        "ROC AUC": roc_auc_score(y_te, s),
        "Brier": brier_score_loss(y_te, s),
    }
    for part in (0.10, 0.20, 0.30):
        c = ciblage_top(y_te, s, part)
        out[f"rappel@top{int(100*part)}%"] = c["rappel"]
        out[f"lift@top{int(100*part)}%"] = c["lift"]
    return out
