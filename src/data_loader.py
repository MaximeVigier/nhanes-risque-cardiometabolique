"""Téléchargement et mise en cache des fichiers NHANES (.XPT) du CDC.

Chaque fichier est récupéré une seule fois puis relu depuis ``data/raw/``.
Les fichiers .XPT (format SAS Transport) se lisent directement avec
``pandas.read_sas`` — aucun paquet SAS n'est nécessaire.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import CDC_URL_TEMPLATE, COMPONENTS, CYCLES, RAW_DIR


def _local_path(component: str, cycle: str) -> Path:
    suffix = CYCLES[cycle]
    return RAW_DIR / f"{component}_{suffix}.xpt"


def download_component(component: str, cycle: str, *, force: bool = False) -> Path:
    """Télécharge un composant NHANES pour un cycle donné, avec cache local.

    Parameters
    ----------
    component : ex. ``"DEMO"``, ``"DR1TOT"`` (voir ``config.COMPONENTS``).
    cycle : ex. ``"2017-2018"`` (voir ``config.CYCLES``).
    force : re-télécharge même si le fichier local existe.

    Returns
    -------
    Chemin du fichier ``.xpt`` local.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = _local_path(component, cycle)
    if dest.exists() and not force:
        return dest

    year1 = cycle.split("-")[0]
    url = CDC_URL_TEMPLATE.format(year1=year1, component=component, suffix=CYCLES[cycle])
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_component(component: str, cycle: str) -> pd.DataFrame:
    """Charge un composant NHANES en DataFrame (télécharge si absent).

    Ajoute une colonne ``cycle`` pour garder la trace de l'origine après fusion.
    """
    path = download_component(component, cycle)
    df = pd.read_sas(path, format="xport")
    df["cycle"] = cycle
    return df


def load_all(components: list[str] | None = None, cycles: list[str] | None = None) -> dict:
    """Charge plusieurs composants sur plusieurs cycles.

    Returns
    -------
    ``{composant: DataFrame}`` où chaque DataFrame empile les cycles demandés.
    """
    components = components or list(COMPONENTS)
    cycles = cycles or list(CYCLES)
    out: dict[str, pd.DataFrame] = {}
    for comp in components:
        frames = []
        for cyc in cycles:
            try:
                frames.append(load_component(comp, cyc))
            except requests.HTTPError as exc:  # certains composants manquent sur un cycle
                print(f"  ! {comp} indisponible pour {cyc} ({exc.response.status_code})")
        if frames:
            out[comp] = pd.concat(frames, ignore_index=True)
    return out


if __name__ == "__main__":
    demo = load_component("DEMO", "2017-2018")
    print(f"DEMO 2017-2018 : {demo.shape[0]} participants, {demo.shape[1]} colonnes")
