"""
Application constants and configuration values.
"""

import os
import seaborn as sns

# Color palette accessible (avoid red/green for color-blind users)
COLOR_PALETTE = sns.color_palette("tab10", 10).as_hex()

# Maximum points for 3D visualization (performance constraint)
MAX_3D_POINTS = 500

# Path to predefined datasets
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

# Predefined datasets configuration
# Each dataset has: filename, description, recommended_target (for classification), min_features (minimum numeric features)
PREDEFINED_DATASETS = {
    "IRIS": {
        "filename": "IRIS 1.csv",
        "description": "Iris flower dataset - 150 instances, 4 features, 3 classes",
        "target_column": "species",
        "num_instances": 150,
        "num_features": 4,
        "num_classes": 3,
        "has_missing_values": False,
        "recommended_for": ["clustering", "classification"],
    },
    "Breast Cancer": {
        "filename": "breast.csv",
        "description": "Breast cancer Wisconsin - 569 instances, 18 features, 2 classes",
        "target_column": "diagnosis",
        "num_instances": 569,
        "num_features": 18,
        "num_classes": 2,
        "has_missing_values": False,
        "recommended_for": ["clustering", "classification"],
    },
    "Heart Disease": {
        "filename": "heart.csv",
        "description": "Heart disease UCI - 303 instances, 13 features, 2 classes",
        "target_column": "output",
        "num_instances": 303,
        "num_features": 13,
        "num_classes": 2,
        "has_missing_values": False,
        "recommended_for": ["clustering", "classification"],
    },
    "Ecoli": {
        "filename": "ecoli.csv",
        "description": "E.coli protein localization - 336 instances, 7 features, 8 classes",
        "target_column": None,  # Last column is target but unnamed
        "num_instances": 336,
        "num_features": 7,
        "num_classes": 8,
        "has_missing_values": False,
        "recommended_for": ["clustering", "classification"],
    },
    "Hepatitis": {
        "filename": "hepatitis.csv",
        "description": "Hepatitis dataset - 155 instances, 19 features, 2 classes (⚠️ missing values)",
        "target_column": "class",
        "num_instances": 155,
        "num_features": 19,
        "num_classes": 2,
        "has_missing_values": True,
        "recommended_for": ["classification"],  # Not ideal for clustering due to missing values
    },
    "Horse Colic": {
        "filename": "horse-colic.csv",
        "description": "Horse colic dataset - 300 instances, 27 features (⚠️ many missing values)",
        "target_column": "outcome",
        "num_instances": 300,
        "num_features": 27,
        "num_classes": 3,
        "has_missing_values": True,
        "recommended_for": ["classification"],  # Requires preprocessing
    },
}

# Supported clustering algorithms
SUPPORTED_ALGORITHMS = (
    "KMeans",      # Centroid-based partitioning
    "K-Medoids",   # Medoid-based partitioning (PAM)
    "DBSCAN",      # Density-based clustering
    "AGNES",       # Agglomerative Nesting (bottom-up hierarchical)
    "DIANA",       # Divisive Analysis (top-down hierarchical)
)

# Supported classifiers
SUPPORTED_CLASSIFIERS = (
    "k-NN",        # k-Nearest Neighbors
    "Naive Bayes", # Naive Bayes Classifier
    "C4.5",        # Decision Tree (C4.5/ID3)
    "SVM",         # Support Vector Machine
)

# Supported file formats for upload
SUPPORTED_FILE_FORMATS = (".csv", ".xls", ".xlsx")

# ============================================================================
# ALGORITHM CONSTRAINTS AND VALIDATION RULES
# ============================================================================

# Minimum dataset requirements for algorithms
ALGORITHM_CONSTRAINTS = {
    # Clustering algorithms
    "KMeans": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "K-Means nécessite au moins 3 échantillons et des données sans valeurs manquantes",
    },
    "K-Medoids": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "K-Medoids nécessite au moins 3 échantillons et des données sans valeurs manquantes",
    },
    "DBSCAN": {
        "min_samples": 5,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "DBSCAN nécessite au moins 5 échantillons (min_samples par défaut) et des données sans valeurs manquantes",
    },
    "AGNES": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "AGNES nécessite au moins 3 échantillons et des données sans valeurs manquantes",
    },
    "DIANA": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "DIANA nécessite au moins 3 échantillons et des données sans valeurs manquantes",
    },
    # Classification algorithms
    "k-NN": {
        "min_samples": 10,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "k-NN nécessite au moins 10 échantillons, 2 classes, et des données sans valeurs manquantes",
    },
    "Naive Bayes": {
        "min_samples": 5,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "requires_non_negative": True,  # Multinomial NB requires non-negative features
        "description": "Naive Bayes nécessite au moins 5 échantillons et 2 classes",
    },
    "C4.5": {
        "min_samples": 5,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "C4.5 nécessite au moins 5 échantillons et 2 classes",
    },
    "SVM": {
        "min_samples": 10,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "SVM nécessite au moins 10 échantillons et 2 classes",
    },
}

# Visualization constraints
VISUALIZATION_CONSTRAINTS = {
    "2D_scatter": {
        "min_features": 2,
        "description": "La visualisation 2D nécessite au moins 2 features",
    },
    "3D_scatter": {
        "min_features": 3,
        "description": "La visualisation 3D nécessite au moins 3 features",
    },
    "dendrogram": {
        "max_samples": 1000,
        "algorithms": ["AGNES", "DIANA"],
        "description": "Le dendrogramme n'est disponible que pour AGNES et DIANA (max 1000 échantillons)",
    },
    "elbow_plot": {
        "algorithms": ["KMeans", "K-Medoids"],
        "description": "La courbe d'Elbow n'est disponible que pour K-Means et K-Medoids",
    },
    "k_distance_graph": {
        "algorithms": ["DBSCAN"],
        "description": "Le graphique k-distance n'est disponible que pour DBSCAN",
    },
}

# Metrics descriptions for help text (clustering)
METRICS_HELP = {
    "silhouette": "Mesure de cohésion/séparation. Valeur dans [-1,1], plus élevé = meilleure séparation.",
    "calinski_harabasz": "Ratio variance inter/intra-cluster (plus élevé = mieux).",
    "davies_bouldin": "Moyenne similarité intra-/inter-cluster (plus faible = mieux).",
}

# Classification metrics descriptions
CLASSIFICATION_METRICS_HELP = {
    "accuracy": "Proportion des prédictions correctes sur l'ensemble total.",
    "precision": "Proportion de vrais positifs parmi les prédictions positives (TP / (TP + FP)).",
    "recall": "Proportion de vrais positifs parmi les vraies classes positives (TP / (TP + FN)).",
    "f1": "Moyenne harmonique de la précision et du rappel (2 * P * R / (P + R)).",
}

# Default train/test split ratio (80/20)
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_TEST_SIZE = 0.2

# k-NN evaluation range (k = 1 to 10)
KNN_K_RANGE = range(1, 11)
