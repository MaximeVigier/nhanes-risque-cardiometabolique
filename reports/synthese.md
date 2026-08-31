# Synthèse — Risque cardiométabolique et mode de vie (NHANES)

## Question

À démographie donnée, l'alimentation et le mode de vie *déclarés* permettent-ils de
prédire le syndrome métabolique, et quels facteurs pèsent le plus ?

## Données

NHANES 2013-2014, 2015-2016, 2017-2018. Après fusion de 17 composants sur `SEQN` et
application des filtres d'inclusion :

| Étape | Restants |
|-------|---------:|
| Participants (3 cycles) | 29 400 |
| Adultes ≥ 20 ans | 17 057 |
| Hors grossesse | 16 867 |
| Sous-échantillon à jeun (glycémie + triglycérides mesurés) | 7 130 |
| Hors diabète / MCV déjà diagnostiqués | **5 521** |

Cible : syndrome métabolique (NCEP ATP III révisé, ≥ 3 critères sur 5), construite à partir
des mesures + traitements. **Prévalence 32,5 %.** 116 participants non classables (trop de
critères manquants).

## Méthode

- Variables d'entrée : démographie, apports nutritionnels (moyenne de 2 rappels 24 h),
  activité physique, sédentarité, tabac, alcool, sommeil. **Exclusion explicite** des
  variables qui définissent la cible (tour de taille, IMC, tension, bilan lipidique,
  glycémie, HbA1c, traitements) — vérifiée par assertion dans `src/features.py`.
- Pré-traitement dans le pipeline scikit-learn (imputation médiane + indicateur,
  standardisation, one-hot), validation croisée stratifiée 5 plis, jeu de test 25 % mis de
  côté.
- Modèles : régression logistique (référence), random forest, HistGradientBoosting.
- Métrique de suivi : average precision ; on rapporte aussi ROC AUC, Brier, et le lift de
  ciblage.

## Résultats

**Comparaison des modèles (CV, train)**

| Modèle | AP (PR-AUC) | ROC AUC | Brier |
|--------|:-----------:|:-------:|:-----:|
| Régression logistique | 0,471 | 0,677 | 0,227 |
| Random forest | 0,461 | 0,670 | 0,209 |
| Gradient boosting | 0,436 | 0,654 | 0,212 |
| Réseau de neurones (MLP) | 0,436 | 0,646 | 0,236 |

La régression logistique fait aussi bien ou mieux que les modèles d'ensemble et qu'un MLP
Keras (notebook 07) : signal faible et à peu près monotone, rien de plus à capter avec un
modèle non linéaire.

**Apport incrémental des blocs (régression logistique, CV)**

| Bloc | ROC AUC | AP |
|------|:-------:|:--:|
| Démographie seule | 0,665 | 0,448 |
| + alimentation | 0,666 | 0,455 |
| + mode de vie | 0,672 | 0,460 |
| *(repère)* + anthropométrie | 0,817 | 0,634 |

C'est le résultat central. **L'alimentation déclarée n'ajoute quasiment rien** au-dessus de
l'âge et de la catégorie sociale (+0,001 de ROC AUC). Le mode de vie ajoute un peu plus
(+0,007). Le vrai pouvoir prédictif est dans le tour de taille et l'IMC — qui sont à moitié
la définition de la cible.

**Test (jeu mis de côté)** : ROC AUC ≈ 0,65 pour les deux modèles retenus. En ciblant les
20 % jugés les plus à risque, on y trouve ~1,5× plus de cas que dans la population (lift),
soit ~29 % de tous les cas captés.

**Sous-groupes** : performance comparable hommes/femmes ; modèle plus discriminant avant
50 ans (l'âge sature ensuite) ; plus de variabilité par origine, en partie liée à la taille
des sous-groupes dans le test.

## Interprétation

Un résultat en creux, mais net et stable sur trois cycles : deux rappels de 24 h sont un
instrument trop grossier pour voir le lien alimentation → syndrome métabolique, qui passe
par des années d'habitudes et par l'adiposité. SHAP et les coefficients de la régression
logistique s'accordent : l'âge domine, puis sexe et origine, puis la sédentarité, puis
quelques marqueurs alimentaires (fibres, sucres) au signe attendu mais d'amplitude faible.

## Limites

Sous-échantillon à jeun seulement ; pas de plan de sondage dans la modélisation
(volontaire, mais interdit la lecture « population ») ; cible transversale (état, pas
incidence) ; non-réponses d'activité physique traitées comme « aucune activité ».

## Prolongements

Analyse pondérée façon épidémiologie ; cible « pré-syndrome » (1-2 critères) ; passer aux
groupes d'aliments et à un score de qualité alimentaire (type HEI) plutôt qu'aux
macronutriments.

---

*Figures : `results/figures/`. Tables : `results/tables/`. Analyse : `notebooks/01` → `06`.*
