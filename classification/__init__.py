"""
Classification algorithms module.

Provides supervised learning algorithms:
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- C4.5 (Decision Tree)
- Support Vector Machine (SVM)
"""

from .algorithms import run_classification, ClassificationResult
from .knn import run_knn, evaluate_knn_k_range
from .naive_bayes import run_naive_bayes
from .decision_tree import run_c45
from .svm import run_svm
from .metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_all_metrics,
)
