import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="NHANES Risque Cardiométabolique",
    page_icon="❤️",
    layout="centered"
)

# Titre et introduction
st.title("📊 Analyse du risque cardiométabolique avec NHANES")
st.markdown("""
Cette application présente les résultats d'une analyse de prévision du **syndrome métabolique** à partir des données NHANES (2013-2018).

> **Contexte :** À âge et catégorie sociale donnés, l'alimentation déclarée n'ajoute presque rien au pouvoir prédictif — c'est un signal faible et monotone.
""")

# Affichage des métriques clés
st.subheader("🔍 Résultats du modèle")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ROC AUC (CV)", "0.68")
with col2:
    st.metric("ROC AUC (Test)", "0.67")
with col3:
    st.metric("Modèle utilisé", "Régression logistique")

# Figure clé : apport par bloc de variables
st.subheader("📊 Apport incrémental des blocs de variables")
st.image("./results/figures/05_apport_par_bloc.png", caption="À âge et catégorie sociale donnés, l'alimentation déclarée n'ajoute presque rien au pouvoir prédictif.", use_container_width=True)

# Section SHAP si disponible
shap_fig = "./results/figures/05_shap_summary.png"
if st.checkbox("Afficher l'analyse SHAP", value=False):
    try:
        st.image(shap_fig, caption="Importance des variables avec SHAP (en régression logistique)", use_container_width=True)
    except Exception as e:
        st.warning(f"Impossible d'afficher la figure SHAP: {e}")

# Section 'Performance par sous-groupe'
st.subheader("👥 Performance par sous-groupe")
subgroup_fig = "./results/figures/03_prevalence_sous_groupes.png"
if os.path.exists(subgroup_fig):
    st.image(subgroup_fig, caption="Prévalence du syndrome métabolique par sous-groupe (sexe, âge, origine).", use_container_width=True)
else:
    st.info("Figure de performance par sous-groupe non disponible dans ce projet.")

# Partie interactive avec le modèle
st.subheader("🔮 Prédiction avec le modèle")

# Le modèle sauvegardé est un dict {pipeline, X_tr, y_tr, X_te, y_te} —
# on récupère le pipeline sklearn et les données d'entraînement (pour les
# valeurs par défaut des variables non couvertes par le formulaire).
try:
    saved = joblib.load('./models/baseline_logit.joblib')
    pipeline = saved['pipeline']
    X_tr = saved['X_tr']
    medianes = X_tr.median(numeric_only=True)

    st.markdown(
        "Le modèle attend 31 variables (nutrition, mode de vie, démographie). "
        "Ajustez les principales ci-dessous ; les autres sont fixées à leur "
        "médiane observée dans le jeu d'entraînement."
    )

    col1, col2 = st.columns(2)
    with col1:
        sexe = st.selectbox("Sexe", sorted(X_tr["sexe"].dropna().unique()))
        age = st.slider("Âge", 20, 80, 40)
        origine = st.selectbox("Origine", sorted(X_tr["origine"].dropna().unique()))
        education = st.selectbox("Niveau d'éducation", sorted(X_tr["education"].dropna().unique()))
    with col2:
        statut_tabac = st.selectbox("Tabac", sorted(X_tr["statut_tabac"].dropna().unique()))
        activite = st.slider("Activité physique (MET-min/semaine)", 0, 3000, 500, step=50)
        sommeil = st.slider("Sommeil (h/nuit)", 3.0, 12.0, 7.5, step=0.5)
        sedentaire = st.slider("Temps sédentaire (min/jour)", 0, 900, 360, step=30)

    if st.button("Calculer la prédiction"):
        row = medianes.to_dict()
        row.update({
            "sexe": sexe,
            "age": float(age),
            "origine": origine,
            "education": education,
            "statut_tabac": statut_tabac,
            "activite_met_min_sem": float(activite),
            "sommeil_h": float(sommeil),
            "sedentaire_min_j": float(sedentaire),
            "transport_actif": bool(X_tr["transport_actif"].mode()[0]),
            "cycle": X_tr["cycle"].mode()[0],
        })
        input_df = pd.DataFrame([row])[X_tr.columns]
        prediction = pipeline.predict_proba(input_df)[0][1]

        st.metric("Probabilité de syndrome métabolique", f"{prediction:.1%}")
        if prediction > 0.5:
            st.markdown("⚠️ Risque élevé de syndrome métabolique (au sens du modèle).")
        else:
            st.markdown("✅ Risque faible de syndrome métabolique (au sens du modèle).")
        st.caption(
            "Rappel : le modèle a un ROC AUC ≈ 0,68 — utile pour illustrer le pipeline, "
            "pas pour un diagnostic individuel."
        )

except Exception as e:
    st.info(f"Les prédictions ne sont pas disponibles pour le moment ({e}).")

# Conclusion
st.subheader("📝 Conclusion")
st.markdown("""
- Le modèle obtient une **performance modérée** (ROC AUC ≈ 0.68) avec des variables démographiques et de mode de vie.
- À âge et catégorie sociale donnés, l'**alimentation déclarée n'ajoute presque rien** au pouvoir prédictif.
- Cette analyse confirme que le signal est **faible et monotone**, et qu'une prédiction robuste nécessite une approche plus complexe à long terme.

---

💡 *Cette application est une démonstration du projet sur les risques cardiométaboliques à partir de données NHANES. Elle affiche les résultats clés, mais ne reproduit pas l'analyse complète.*""")
st.markdown("---")

# Lien vers le repo
st.markdown("📋 Retrouvez le projet complet sur [GitHub](https://github.com/MaximeVigier/nhanes-risque-cardiometabolique)")