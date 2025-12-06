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
        "description": "K-Means requires at least 3 samples and data without missing values",
    },
    "K-Medoids": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "K-Medoids requires at least 3 samples and data without missing values",
    },
    "DBSCAN": {
        "min_samples": 5,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "DBSCAN requires at least 5 samples (default min_samples) and data without missing values",
    },
    "AGNES": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "AGNES requires at least 3 samples and data without missing values",
    },
    "DIANA": {
        "min_samples": 3,
        "min_features": 1,
        "requires_no_missing": True,
        "description": "DIANA requires at least 3 samples and data without missing values",
    },
    # Classification algorithms
    "k-NN": {
        "min_samples": 10,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "k-NN requires at least 10 samples, 2 classes, and data without missing values",
    },
    "Naive Bayes": {
        "min_samples": 5,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "requires_non_negative": True,  # Multinomial NB requires non-negative features
        "description": "Naive Bayes requires at least 5 samples and 2 classes",
    },
    "C4.5": {
        "min_samples": 5,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "C4.5 requires at least 5 samples and 2 classes",
    },
    "SVM": {
        "min_samples": 10,
        "min_features": 1,
        "min_classes": 2,
        "requires_no_missing": True,
        "description": "SVM requires at least 10 samples and 2 classes",
    },
}

# Visualization constraints
VISUALIZATION_CONSTRAINTS = {
    "2D_scatter": {
        "min_features": 2,
        "description": "2D visualization requires at least 2 features",
    },
    "3D_scatter": {
        "min_features": 3,
        "description": "3D visualization requires at least 3 features",
    },
    "dendrogram": {
        "max_samples": 1000,
        "algorithms": ["AGNES", "DIANA"],
        "description": "Dendrogram is only available for AGNES and DIANA (max 1000 samples)",
    },
    "elbow_plot": {
        "algorithms": ["KMeans", "K-Medoids"],
        "description": "Elbow curve is only available for K-Means and K-Medoids",
    },
    "k_distance_graph": {
        "algorithms": ["DBSCAN"],
        "description": "k-distance graph is only available for DBSCAN",
    },
}

# Metrics descriptions for help text (clustering)
METRICS_HELP = {
    "silhouette": "Cohesion/separation measure. Value in [-1,1], higher = better separation.",
    "calinski_harabasz": "Ratio of between-cluster to within-cluster dispersion (higher = better).",
    "davies_bouldin": "Average similarity between clusters (lower = better).",
}

# Classification metrics descriptions
CLASSIFICATION_METRICS_HELP = {
    "accuracy": "Proportion of correct predictions over the total set.",
    "precision": "Proportion of true positives among positive predictions (TP / (TP + FP)).",
    "recall": "Proportion of true positives among actual positive instances (TP / (TP + FN)).",
    "f1": "Harmonic mean of precision and recall (2 * P * R / (P + R)).",
}

# Default train/test split ratio (80/20)
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_TEST_SIZE = 0.2

# k-NN evaluation range (k = 1 to 10)
KNN_K_RANGE = range(1, 11)
