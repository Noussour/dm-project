# 🔬 Comparateur d'Algorithmes de Clustering & Classification

> Application pédagogique de **Data Mining** (Semestre 7)

Une application web interactive construite avec **Streamlit** permettant de comparer différents algorithmes de clustering et de classification sur des jeux de données personnalisés.

---

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Algorithmes Supportés](#-algorithmes-supportés)
- [Architecture du Projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Métriques d'Évaluation](#-métriques-dévaluation)
- [Docker](#-docker)
- [Technologies Utilisées](#-technologies-utilisées)

---

## ✨ Fonctionnalités

- **Chargement de données** : Upload de fichiers CSV/Excel
- **Prétraitement des données** : Gestion des valeurs manquantes, outliers, normalisation
- **Sélection de features** : Choix interactif des colonnes numériques
- **Clustering** : K-Means, K-Medoids, DBSCAN, AGNES, DIANA
- **Classification** : k-NN, Naive Bayes, C4.5, SVM
- **Visualisation 2D/3D** : Graphiques interactifs Plotly
- **Métriques détaillées** : Scores Silhouette, Calinski-Harabasz, Davies-Bouldin (clustering), Précision, Rappel, F-mesure (classification)
- **Comparaison** : Comparer tous les algorithmes en un clic

---

## 🤖 Algorithmes Supportés

### Clustering

| Algorithme | Description | Paramètres |
|------------|-------------|------------|
| **K-Means** | Partitionnement en k clusters basé sur les centroïdes | `n_clusters`, `init` |
| **K-Medoids** | Partitionnement robuste aux outliers (PAM) | `n_clusters`, `metric` |
| **DBSCAN** | Clustering basé sur la densité, détecte le bruit | `eps`, `min_samples` |
| **AGNES** | Agglomerative Nesting (approche ascendante) | `n_clusters`, `linkage` |
| **DIANA** | Divisive Analysis (approche descendante) | `n_clusters`, `metric` |

### Classification

| Algorithme | Description | Paramètres |
|------------|-------------|------------|
| **k-NN** | k plus proches voisins | `k`, `metric`, `weights` |
| **Naive Bayes** | Classifieur bayésien naïf | `type` (gaussian, multinomial) |
| **C4.5** | Arbre de décision (gain d'information) | `criterion`, `max_depth` |
| **SVM** | Machine à vecteurs de support | `kernel`, `C`, `gamma` |

---

## 📁 Architecture du Projet

```
TP4/
├── app.py                  # Point d'entrée principal
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Image Docker
├── docker-compose.yaml     # Orchestration Docker
│
├── config/                 # Configuration et constantes
│   ├── __init__.py
│   ├── settings.py         # Configuration Streamlit
│   └── constants.py        # Couleurs, algorithmes supportés, limites
│
├── utils/                  # Fonctions utilitaires
│   ├── __init__.py
│   ├── data_loader.py      # Chargement/validation des données
│   ├── preprocessing.py    # Pipeline de prétraitement
│   └── metrics.py          # Métriques de clustering
│
├── clustering/             # Implémentations des algorithmes de clustering
│   ├── __init__.py
│   ├── algorithms.py       # Orchestrateur principal
│   ├── kmeans.py           # K-Means
│   ├── kmedoids.py         # K-Medoids (PAM)
│   ├── dbscan.py           # DBSCAN
│   ├── agnes.py            # AGNES
│   └── diana.py            # DIANA
│
├── classification/         # Implémentations des algorithmes de classification
│   ├── __init__.py
│   ├── algorithms.py       # Orchestrateur principal
│   ├── knn.py              # k-NN
│   ├── naive_bayes.py      # Naive Bayes
│   ├── decision_tree.py    # C4.5
│   ├── svm.py              # SVM
│   └── metrics.py          # Métriques de classification
│
├── visualization/          # Visualisation des résultats
│   ├── __init__.py
│   ├── plots.py            # Graphiques
│   └── colors.py           # Gestion palette de couleurs
│
└── components/             # Composants UI Streamlit
    ├── __init__.py
    ├── sidebar.py          # Sidebar clustering
    ├── classification_sidebar.py  # Sidebar classification
    ├── tabs.py             # Onglets clustering
    └── classification_tabs.py  # Onglets classification
```

---

## 🚀 Installation

### Prérequis

- Python 3.11+
- pip

### Installation locale

```bash
# 1. Cloner ou accéder au répertoire
cd TP4

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application sera accessible à l'adresse : **http://localhost:8501**

---

## 💻 Utilisation

### 1. Charger les données

Glisser-déposer un fichier CSV ou Excel dans la zone de téléchargement.

### 2. Prétraitement

Utilisez l'onglet **Prétraitement** pour :
- Supprimer les outliers (méthode IQR ou Z-score)
- Gérer les valeurs manquantes (suppression, moyenne, médiane)
- Normaliser les données (Min-Max, Z-score)

### 3. Clustering

1. Sélectionner l'algorithme (K-Means, K-Medoids, DBSCAN, AGNES, DIANA)
2. Ajuster les paramètres spécifiques
3. Cliquer sur **"Exécuter"** ou **"Meilleurs params"**

### 4. Classification

1. Sélectionner la variable cible (classe)
2. Configurer le partitionnement (80% apprentissage, 20% test)
3. Choisir l'algorithme (k-NN, Naive Bayes, C4.5, SVM)
4. Cliquer sur **"Classifier"** ou **"Comparer tous"**

---

## 📊 Métriques d'Évaluation

### Clustering

| Métrique | Plage | Interprétation |
|----------|-------|----------------|
| **Silhouette Score** | [-1, 1] | Plus élevé = meilleure séparation entre clusters |
| **Calinski-Harabasz** | [0, +∞) | Plus élevé = clusters plus denses et bien séparés |
| **Davies-Bouldin** | [0, +∞) | Plus faible = meilleure distinction entre clusters |

### Classification

| Métrique | Description |
|----------|-------------|
| **Précision** | TP / (TP + FP) |
| **Rappel** | TP / (TP + FN) |
| **F-mesure** | 2 × (P × R) / (P + R) |
| **Matrice de Confusion** | TP, TN, FP, FN |

---

## 🐳 Docker

### Lancer avec Docker Compose

```bash
# Construire et lancer
docker-compose up --build

# En mode détaché
docker-compose up -d --build
```

### Lancer avec Docker directement

```bash
# Construire l'image
docker build -t tp4-clustering .

# Lancer le conteneur
docker run -p 8501:8501 tp4-clustering
```

Accéder à l'application : **http://localhost:8501**

---

## 🛠 Technologies Utilisées

| Technologie | Rôle |
|-------------|------|
| **Streamlit** | Framework web interactif |
| **scikit-learn** | Algorithmes de clustering, classification et métriques |
| **Plotly** | Visualisations interactives 2D/3D |
| **Matplotlib/Seaborn** | Graphiques et palettes de couleurs |
| **Pandas/NumPy** | Manipulation des données |
| **SciPy** | Clustering hiérarchique (linkage) |

---

## 📝 Notes Pédagogiques

Cette application a été développée dans le cadre du **TP4 de Data Mining** pour permettre aux étudiants de :

1. **Comprendre** les différences entre algorithmes de clustering
2. **Visualiser** l'impact des paramètres sur les résultats
3. **Comparer** objectivement les algorithmes via des métriques standardisées
4. **Explorer** différents types de données (réelles et synthétiques)