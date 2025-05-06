# 🎮 Battery Market Predictor

![Version](https://img.shields.io/badge/version-1.0.0-yellow)
![Python](https://img.shields.io/badge/Python-3.8+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-coral)

<p align="center">
  <img src="https://i.imgur.com/PLACEHOLDER.png" alt="Battery Market Predictor Screenshot" width="850">
  <br>
  <em>Analyse prédictive du marché des services système pour le stockage par batteries en France</em>
</p>

## 🔋 Qu'est-ce que c'est?

![image](https://github.com/user-attachments/assets/fdb7f476-860b-4555-b899-5902cdd85a66)

![image](https://github.com/user-attachments/assets/8d03f045-3daa-43ff-bbc1-b5e646c72d03)

![image](https://github.com/user-attachments/assets/cbf29f71-cbe2-46ec-91c9-f020fac3b153)


**Battery Market Predictor** est une application Streamlit au style rétro-gaming conçue pour analyser et prédire les dynamiques du marché des services système en France, spécifiquement pour l'optimisation des systèmes de stockage d'énergie par batteries.

Cette application représente un MVP (Produit Minimum Viable) qui permet aux équipes commerciales et d'exploitation de Qair de:
- Visualiser les tendances historiques des prix FCR et aFRR
- Prédire les prix de compensation et probabilités d'activation
- Optimiser les stratégies d'opération des batteries
- Évaluer la rentabilité financière des actifs de stockage

## 🎯 Fonctionnalités

### 📊 Tableau de Bord du Marché
- Visualisation des tendances de prix des réserves primaires (FCR) et secondaires (aFRR)
- Analyse des patterns de prix par jour et par heure
- Suivi des métriques clés du marché

### 🔮 Module de Prédiction
- Prévision des prix pour FCR et aFRR
- Intervalles de confiance pour les prédictions
- Évaluation de la précision des prédictions

### 💰 Économie des Batteries
- Calcul des revenus potentiels pour les actifs de stockage
- Optimisation de la stratégie d'opération des batteries
- Analyse des métriques d'investissement (VAN, TRI, période d'amortissement)

### 🧪 Explorateur de Données
- Exploration des données brutes et traitées
- Visualisation des statistiques descriptives
- Analyse des corrélations entre variables de marché

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/qair/battery-market-predictor.git
cd battery-market-predictor

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

## 🎨 Design Rétro-Gaming

L'application présente une esthétique rétro-gaming distinctive:
- Interface pixélisée avec contours nets et ombres en bloc
- Palette de couleurs vibrantes dominée par le jaune doré et le corail
- Typographie inspirée des jeux vidéo avec les polices VT323 et Space Mono
- Éléments interactifs et animations rappelant les jeux vidéo classiques

## 💼 Pourquoi c'est important pour Qair

Ce projet répond à des besoins stratégiques essentiels pour Qair:

1. **Création de nouvelles sources de revenus** - En optimisant la participation aux marchés des services système, Qair peut diversifier ses revenus au-delà de la simple vente d'électricité.

2. **Avantage concurrentiel** - La compréhension approfondie des dynamiques de marché offre un avantage stratégique dans le développement de projets de stockage.

3. **Transition énergétique** - Le stockage par batteries est crucial pour l'intégration des énergies renouvelables dans le réseau, alignant ce projet avec la mission fondamentale de Qair.

4. **Valorisation des actifs** - L'optimisation de la participation aux services système augmente significativement la valeur des projets de stockage existants et futurs.

5. **Aide à la décision** - Fournit aux équipes commerciales et techniques des outils visuels intuitifs pour prendre des décisions éclairées rapidement.

## 🌐 Architecture

```
ancillary_services_predictor/
│
├── app.py                  # Point d'entrée de l'application Streamlit
├── requirements.txt        # Dépendances
│
├── data/                   # Répertoire de données
│   └── sample_data.csv     # Jeu de données exemple pour tests
│
├── src/                    # Code source
│   ├── __init__.py
│   ├── data_processing.py  # Fonctions de chargement et traitement des données
│   ├── visualization.py    # Fonctions de visualisation
│   ├── prediction.py       # Modèles de prédiction
│   └── styles.py           # Stylisation UI et CSS personnalisé
│
└── README.md               # Documentation du projet
```

## 🔄 Utilisation des Données

L'application peut fonctionner avec:
- Des données générées automatiquement (à des fins de démonstration)
- Des fichiers CSV téléchargés par l'utilisateur contenant des prix de marché et données réseau

## 🔍 Prochaines Étapes

- Intégration des API de données des gestionnaires de réseau (RTE)
- Ajout de modèles de prédiction plus avancés (LSTM, transformers)
- Optimisation multi-marchés (arbitrage + services système)
- Interface mobile pour suivi en temps réel

## 📜 Licence

Ce projet est sous licence interne Qair - tous droits réservés.

---

<p align="center">
  <em>Développé par l'équipe Data Science de Qair - 2025</em>
</p>
