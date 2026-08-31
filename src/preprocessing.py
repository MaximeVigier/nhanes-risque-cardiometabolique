"""Nettoyage des composants NHANES et assemblage de la table analytique.

Chaque bloc (démographie, examen, biologie, alimentation, mode de vie,
antécédents) est extrait séparément, indexé par ``SEQN``, puis tout est joint sur
la table démographique.

Deux points demandent un peu de soin :

* le codage des non-réponses (7/9/77/99…) diffère d'une variable à l'autre —
  géré via ``config.SPECIAL_MISSING`` ;
* l'alcool et le sommeil changent de questionnaire entre 2013-2016 et 2017-2018 —
  harmonisés ci-dessous, avec les hypothèses écrites noir sur blanc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _blank_special(df: pd.DataFrame) -> pd.DataFrame:
    """Passe en NaN les codes « Refus / Ne sait pas » listés dans la config."""
    df = df.copy()
    for col, codes in config.SPECIAL_MISSING.items():
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(codes))
    return df


def _fix_xpt_zeros(s: pd.Series, atol: float = 1e-12) -> pd.Series:
    """Certaines valeurs 0 ressortent en ~5e-79 après lecture du format XPT.
    On les recale à 0."""
    return s.mask(s.abs() < atol, 0.0)


def _index_seqn(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(SEQN=df["SEQN"].astype("int64")).set_index("SEQN")


# --------------------------------------------------------------------------- #
# Blocs
# --------------------------------------------------------------------------- #
def demographie(demo: pd.DataFrame) -> pd.DataFrame:
    d = _blank_special(_index_seqn(demo))
    out = pd.DataFrame(index=d.index)
    out["cycle"] = d["cycle"]
    out["age"] = d["RIDAGEYR"]
    out["sexe"] = d["RIAGENDR"].map(config.SEXE)
    out["origine"] = d["RIDRETH3"].map(config.ORIGINE)
    out["education"] = d["DMDEDUC2"].map(config.EDUCATION)
    out["ratio_revenu"] = d["INDFMPIR"]            # revenu familial / seuil de pauvreté
    out["enceinte"] = d.get("RIDEXPRG").eq(1) if "RIDEXPRG" in d else False
    # pondérations d'enquête (utilisées seulement en descriptif, pas pour le modèle)
    out["poids_mec"] = d["WTMEC2YR"]
    out["strate"] = d["SDMVSTRA"]
    out["psu"] = d["SDMVPSU"]
    return out


def examen(bmx: pd.DataFrame, bpx: pd.DataFrame) -> pd.DataFrame:
    b = _index_seqn(bmx)
    p = _index_seqn(bpx)
    out = pd.DataFrame(index=b.index.union(p.index))
    out["tour_taille_cm"] = b["BMXWAIST"]
    out["imc"] = b["BMXBMI"]
    # tension : moyenne des mesures disponibles (NHANES en fait jusqu'à 4)
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in p]
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in p]
    out["pas"] = p[sys_cols].replace(0, np.nan).mean(axis=1)
    out["pad"] = p[dia_cols].replace(0, np.nan).mean(axis=1)
    return out


def biologie(ghb, glu, hdl, trigly, tchol) -> pd.DataFrame:
    g = _index_seqn(ghb); gl = _index_seqn(glu)
    h = _index_seqn(hdl); tr = _index_seqn(trigly); tc = _index_seqn(tchol)
    idx = g.index.union(gl.index).union(h.index).union(tr.index).union(tc.index)
    out = pd.DataFrame(index=idx)
    out["hba1c"] = g["LBXGH"]
    out["glycemie_jeun"] = gl["LBXGLU"]
    out["hdl"] = h["LBDHDD"]
    out["triglycerides"] = tr["LBXTR"]
    out["cholesterol_total"] = tc["LBXTC"]
    out["a_jeun"] = out["glycemie_jeun"].notna() & out["triglycerides"].notna()
    return out


def alimentation(dr1: pd.DataFrame, dr2: pd.DataFrame) -> pd.DataFrame:
    """Apports nutritionnels : moyenne des deux rappels 24 h quand ils sont
    tous les deux valides (statut de rappel = 1), sinon le jour 1 seul."""
    keep = {
        "KCAL": "energie_kcal", "PROT": "proteines_g", "CARB": "glucides_g",
        "SUGR": "sucres_g", "FIBE": "fibres_g", "TFAT": "lipides_g",
        "SFAT": "ags_g", "MFAT": "agmi_g", "PFAT": "agpi_g",
        "CHOL": "cholesterol_alim_mg", "SODI": "sodium_mg", "POTA": "potassium_mg",
        "CAFF": "cafeine_mg", "ALCO": "alcool_alim_g",
    }
    d1 = _index_seqn(dr1); d2 = _index_seqn(dr2)
    v1 = pd.DataFrame({dst: d1.get(f"DR1T{src}") for src, dst in keep.items()}, index=d1.index)
    v1 = v1[d1["DR1DRSTZ"].eq(1)]  # rappel jour 1 « fiable et complet »
    v2 = pd.DataFrame({dst: d2.get(f"DR2T{src}") for src, dst in keep.items()}, index=d2.index)
    v2 = v2[d2["DR2DRSTZ"].eq(1)] if "DR2DRSTZ" in d2 else v2.iloc[0:0]

    out = v1.copy()
    common = out.index.intersection(v2.index)
    out.loc[common] = (v1.loc[common] + v2.loc[common]) / 2
    out["n_rappels"] = 1
    out.loc[common, "n_rappels"] = 2

    # quelques dérivés utiles et interprétables
    out["densite_energetique"] = out["energie_kcal"]  # kcal ; densité /100g nécessiterait le poids d'aliments
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_lipides"] = 9 * out["lipides_g"] / out["energie_kcal"] * 100
        out["pct_glucides"] = 4 * out["glucides_g"] / out["energie_kcal"] * 100
        out["pct_proteines"] = 4 * out["proteines_g"] / out["energie_kcal"] * 100
        out["ratio_agpi_ags"] = out["agpi_g"] / out["ags_g"]
        out["sucres_pct_kcal"] = 4 * out["sucres_g"] / out["energie_kcal"] * 100
        out["fibres_pour_1000kcal"] = out["fibres_g"] / out["energie_kcal"] * 1000
        out["sodium_pour_1000kcal"] = out["sodium_mg"] / out["energie_kcal"] * 1000
    return out.replace([np.inf, -np.inf], np.nan)


def _drinks_per_week_1316(d: pd.DataFrame) -> pd.Series:
    """Verres/semaine estimés, questionnaire 2013-2016 (ALQ101/120Q/120U/130)."""
    unit = {1: 1.0, 2: 12 / 52, 3: 1 / 52}  # semaine / mois / an -> par semaine
    freq = d["ALQ120Q"] * d["ALQ120U"].map(unit)
    dpw = freq * d["ALQ130"]
    dpw = dpw.where(d["ALQ101"].ne(2), 0.0)     # « jamais 12 verres/an » -> 0
    return dpw


def _drinks_per_week_1718(d: pd.DataFrame) -> pd.Series:
    """Verres/semaine estimés, questionnaire 2017-2018 (ALQ111/121/130).
    ALQ121 est une fréquence codée ; on la convertit en occasions/an."""
    per_year = {
        0: 0, 1: 365, 2: 300, 3: 182, 4: 104, 5: 52,
        6: 30, 7: 12, 8: 9, 9: 4.5, 10: 1.5,
    }
    occ_week = d["ALQ121"].map(per_year) / 52
    dpw = occ_week * d["ALQ130"]
    dpw = dpw.where(d["ALQ111"].ne(2), 0.0)
    return dpw


def mode_de_vie(paq, smq, alq, slq) -> pd.DataFrame:
    pa = _blank_special(_index_seqn(paq))
    sm = _blank_special(_index_seqn(smq))
    al = _blank_special(_index_seqn(alq))
    sl = _blank_special(_index_seqn(slq))
    idx = pa.index.union(sm.index).union(al.index).union(sl.index)
    out = pd.DataFrame(index=idx)

    # --- activité physique : MET-min/semaine (loisir), transport actif, sédentarité
    vig = (pa["PAQ650"].eq(1) * pa["PAQ655"].fillna(0) * pa["PAD660"].fillna(0))
    mod = (pa["PAQ665"].eq(1) * pa["PAQ670"].fillna(0) * pa["PAD675"].fillna(0))
    out["activite_met_min_sem"] = 8 * vig + 4 * mod
    out["sedentaire_min_j"] = pa["PAD680"]
    out["transport_actif"] = pa["PAQ635"].eq(1)

    # --- tabac
    jamais = sm["SMQ020"].eq(2)
    fumeur = sm["SMQ040"].isin([1, 2])
    ancien = sm["SMQ020"].eq(1) & sm["SMQ040"].eq(3)
    statut = pd.Series(pd.NA, index=sm.index, dtype="object")
    statut[jamais] = "Jamais"
    statut[ancien] = "Ancien"
    statut[fumeur] = "Fumeur"
    out["statut_tabac"] = statut

    # --- alcool (harmonisation des deux questionnaires)
    dpw = pd.Series(np.nan, index=al.index)
    m_old = al["cycle"].isin(["2013-2014", "2015-2016"])
    if "ALQ101" in al:
        dpw.loc[m_old] = _drinks_per_week_1316(al.loc[m_old])
    if "ALQ121" in al:
        dpw.loc[~m_old] = _drinks_per_week_1718(al.loc[~m_old])
    out["alcool_verres_sem"] = _fix_xpt_zeros(dpw).clip(lower=0)

    # --- sommeil
    h = sl["SLD010H"] if "SLD010H" in sl else pd.Series(np.nan, index=sl.index)
    h2 = sl["SLD012"] if "SLD012" in sl else pd.Series(np.nan, index=sl.index)
    out["sommeil_h"] = h.combine_first(h2)
    return out


def antecedents(diq, bpq, mcq) -> pd.DataFrame:
    di = _blank_special(_index_seqn(diq))
    bp = _blank_special(_index_seqn(bpq))
    mc = _blank_special(_index_seqn(mcq))
    idx = di.index.union(bp.index).union(mc.index)
    out = pd.DataFrame(index=idx)
    out["diabete_declare"] = di["DIQ010"].eq(1)
    out["traite_diabete"] = di["DIQ050"].eq(1) | di["DIQ070"].eq(1)
    out["traite_hta"] = bp["BPQ050A"].eq(1)
    out["traite_chol"] = bp["BPQ100D"].eq(1)
    mi = [c for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"] if c in mc]
    out["mcv_declaree"] = mc[mi].eq(1).any(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def assemble(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Joint tous les blocs sur la table démographique (jointure à gauche)."""
    demo = demographie(tables["DEMO"])
    blocs = [
        examen(tables["BMX"], tables["BPX"]),
        biologie(tables["GHB"], tables["GLU"], tables["HDL"],
                 tables["TRIGLY"], tables["TCHOL"]),
        alimentation(tables["DR1TOT"], tables["DR2TOT"]),
        mode_de_vie(tables["PAQ"], tables["SMQ"], tables["ALQ"], tables["SLQ"]),
        antecedents(tables["DIQ"], tables["BPQ"], tables["MCQ"]),
    ]
    df = demo
    for b in blocs:
        df = df.join(b, how="left")
    return df


def filtre_population(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique les filtres d'inclusion et renvoie (table filtrée, journal des pertes)."""
    log = []
    n = len(df)
    log.append(("départ", n, n))

    step = df[df["age"] >= config.AGE_MIN]
    log.append((f"âge ≥ {config.AGE_MIN} ans", n - len(step), len(step))); n = len(step)

    step = step[~step["enceinte"].fillna(False)]
    log.append(("hors grossesse", n - len(step), len(step))); n = len(step)

    step = step[step["a_jeun"].fillna(False)]
    log.append(("sous-échantillon à jeun (glycémie + TG)", n - len(step), len(step))); n = len(step)

    step = step[~(step["diabete_declare"].fillna(False) | step["traite_diabete"].fillna(False)
                  | step["mcv_declaree"].fillna(False))]
    log.append(("hors diabète / MCV déjà diagnostiqués", n - len(step), len(step))); n = len(step)

    journal = pd.DataFrame(log, columns=["étape", "exclus", "restants"])
    return step, journal
