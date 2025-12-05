"""
Application constants and configuration values.
"""

import seaborn as sns

# Color palette accessible (avoid red/green for color-blind users)
COLOR_PALETTE = sns.color_palette("tab10", 10).as_hex()

# Maximum points for 3D visualization (performance constraint)
MAX_3D_POINTS = 500

# Supported clustering algorithms
SUPPORTED_ALGORITHMS = (
    "KMeans",      # Centroid-based partitioning
    "K-Medoids",   # Medoid-based partitioning (PAM)
    "DBSCAN",      # Density-based clustering
    "AGNES",       # Agglomerative Nesting (bottom-up hierarchical)
    "DIANA",       # Divisive Analysis (top-down hierarchical)
)

# Supported classification algorithms
SUPPORTED_CLASSIFIERS = (
    "k-NN",        # k-Nearest Neighbors
    "Naive Bayes", # Naive Bayes Classifier
    "C4.5",        # Decision Tree (C4.5/ID3)
    "SVM",         # Support Vector Machine
)

# Supported file formats for upload
SUPPORTED_FILE_FORMATS = (".csv", ".xls", ".xlsx")

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
