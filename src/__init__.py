"""Modules réutilisables du projet Risque cardiométabolique / NHANES.

Ordre logique d'utilisation :
    config        -> chemins, cycles, correspondances de fichiers, seuils NCEP
    data_loader   -> téléchargement + cache des fichiers .XPT du CDC
    preprocessing -> recodage du manquant, fusion sur SEQN, harmonisation des cycles
    target        -> construction de la cible « syndrome métabolique » (NCEP ATP III)
    features      -> assemblage de la matrice de features (sans fuite de données)
    modeling      -> pipelines scikit-learn, validation croisée, métriques
    plotting      -> figures d'exploration et de résultats
"""
