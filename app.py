import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page Streamlit
st.set_page_config(
    page_title="NHANES Risque Cardiométabolique",
    page_icon=" heart",
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
st.image("./results/figures/06_performance_sous_groupes.png", caption="Performance du modèle en fonction du sexe, de l'âge et de l'origine.", use_container_width=True)

# Partie interactive avec le modèle
st.subheader("🔮 Prédiction avec le modèle")

# Vérifions si le modèle est utilisable
try:
    model = joblib.load('./models/baseline_logit.joblib')
    feature_names = model.feature_names_in_
    
    # Formulaire interactif simple
    st.markdown("Saisissez quelques caractéristiques démographiques et de mode de vie pour obtenir une prédiction :")
    
    # Exemple de variables (selon les données NHANES)
    gender = st.selectbox("Sexe", ["Homme", "Femme"])
    age = st.slider("Âge", 20, 80, 40)
    race = st.selectbox("Race/origine", ["Blanc", "Noir", "Asiatique", "Autre"])
    education = st.selectbox("Niveau d'éducation", ["Moins de 12 ans", "12 ans", "Plus de 12 ans"])
    smoking = st.selectbox("Fumeur", ["Non", "Ancien", "Actuel"])
    
    # Conversion en variables numériques
    gender_val = 1 if gender == "Homme" else 0
    race_map = {"Blanc": 0, "Noir": 1, "Asiatique": 2, "Autre": 3}
    race_val = race_map[race]
    
    edu_val = {"Moins de 12 ans": 0, "12 ans": 1, "Plus de 12 ans": 2}
    edu_val = edu_val[education]
    
    smoke_map = {"Non": 0, "Ancien": 1, "Actuel": 2}
    smoke_val = smoke_map[smoking]
    
    if st.button("Calculer la prédiction"):
        # On utilise une instance de données fictive pour tester le modèle
        input_data = np.array([gender_val, age, race_val, edu_val, smoke_val, 0.5, 0.3, 0.1])  
        prediction = model.predict_proba(input_data.reshape(1, -1))[0][1]
        
        st.metric("Probabilité de syndrome métabolique", f"{prediction:.2%}")
        if prediction > 0.5:
            st.markdown("⚠️ Risque élevé de syndrome métabolique.")
        else:
            st.markdown("✅ Risque faible de syndrome métabolique.")
            
except Exception as e:
    st.info("Les prédictions ne sont pas disponibles pour le moment - chargement du modèle échoué.")

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