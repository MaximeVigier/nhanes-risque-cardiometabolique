"""Configuration centrale : chemins, cycles NHANES, fichiers à récupérer,
codage des variables et seuils cliniques.

Rien de « métier » ici, juste des constantes partagées par les notebooks et les
autres modules.
"""
from __future__ import annotations

from pathlib import Path

# Racine du projet = dossier parent de src/. Permet d'appeler les modules depuis
# les notebooks sans se soucier du dossier courant.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------- #
# Cycles NHANES et suffixe de fichier associé
# --------------------------------------------------------------------------- #
# NHANES nomme chaque fichier <COMPOSANT>_<SUFFIXE>.xpt, un suffixe par cycle.
CYCLES: dict[str, str] = {
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}

# URL des fichiers publics (vérifiée en août 2026) :
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<annee1>/DataFiles/<COMPOSANT>_<SUFFIXE>.xpt
CDC_URL_TEMPLATE = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year1}/DataFiles/{component}_{suffix}.xpt"
)

# Composants à télécharger, avec un mot sur le rôle de chacun.
COMPONENTS: dict[str, str] = {
    "DEMO": "âge, sexe, origine, revenu, éducation, pondérations",
    "BMX": "mesures anthropométriques (tour de taille, IMC)",
    "BPX": "tension artérielle (plusieurs mesures)",
    "GHB": "hémoglobine glyquée (HbA1c)",
    "GLU": "glycémie à jeun (sous-échantillon)",
    "HDL": "HDL-cholestérol",
    "TRIGLY": "triglycérides (sous-échantillon à jeun)",
    "TCHOL": "cholestérol total",
    "DR1TOT": "rappel alimentaire 24 h, jour 1",
    "DR2TOT": "rappel alimentaire 24 h, jour 2",
    "PAQ": "activité physique",
    "SMQ": "tabac",
    "ALQ": "alcool",
    "SLQ": "sommeil",
    "DIQ": "diabète déclaré / traitement",
    "BPQ": "HTA et cholestérol déclarés / traitement",
    "MCQ": "antécédents cardiovasculaires déclarés",
}

# --------------------------------------------------------------------------- #
# Codage des variables catégorielles (source : documentation NHANES)
# --------------------------------------------------------------------------- #
SEXE = {1: "Homme", 2: "Femme"}

ORIGINE = {
    1: "Hispanique (Mexique)",
    2: "Hispanique (autre)",
    3: "Blanc non hispanique",
    4: "Noir non hispanique",
    6: "Asiatique non hispanique",
    7: "Autre / multiracial",
}

EDUCATION = {
    1: "< collège",
    2: "Collège sans diplôme",
    3: "Bac / équivalent",
    4: "Études sup. courtes",
    5: "Diplôme universitaire",
}

# Valeurs « Refus » / « Ne sait pas » à passer en NaN, par variable.
SPECIAL_MISSING: dict[str, tuple[int, ...]] = {
    "DMDEDUC2": (7, 9),
    "DIQ010": (7, 9),
    "DIQ050": (7, 9),
    "DIQ070": (7, 9),
    "BPQ040A": (7, 9),
    "BPQ050A": (7, 9),
    "BPQ090D": (7, 9),
    "BPQ100D": (7, 9),
    "MCQ160B": (7, 9),
    "MCQ160C": (7, 9),
    "MCQ160E": (7, 9),
    "MCQ160F": (7, 9),
    "SMQ020": (7, 9),
    "SMQ040": (7, 9),
    "PAQ650": (7, 9),
    "PAQ665": (7, 9),
    "PAD660": (7777, 9999),
    "PAD675": (7777, 9999),
    "PAD680": (7777, 9999),
    "ALQ121": (77, 99),
    "ALQ130": (777, 999),
    "ALQ101": (7, 9),
    "ALQ120Q": (777, 999),
    "SLD010H": (77, 99),
    "SLD012": (77, 99),
}

# --------------------------------------------------------------------------- #
# Syndrome métabolique — critères NCEP ATP III révisés (Grundy 2005 / Alberti 2009)
# ≥ 3 critères sur 5 = syndrome métabolique.
# --------------------------------------------------------------------------- #
NCEP = {
    "tour_taille_cm": {"Homme": 102.0, "Femme": 88.0},   # critère si AU-DESSUS
    "triglycerides": 150.0,                               # mg/dL, au-dessus
    "hdl": {"Homme": 40.0, "Femme": 50.0},               # mg/dL, EN DESSOUS
    "pas": 130.0,                                         # mmHg systolique
    "pad": 85.0,                                          # mmHg diastolique
    "glycemie_jeun": 100.0,                               # mg/dL, au-dessus
}

# Variables qui servent à définir la cible : interdites en entrée du modèle.
LEAKAGE = [
    "tour_taille_cm", "imc", "triglycerides", "hdl", "cholesterol_total",
    "pas", "pad", "glycemie_jeun", "hba1c",
    "traite_hta", "traite_chol", "traite_diabete",
    "diabete_declare",
]

AGE_MIN = 20  # adultes
