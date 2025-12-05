"""
C4.5 Decision Tree classifier implementation.

C4.5 is a decision tree algorithm developed by Ross Quinlan that uses
information gain ratio for feature selection. In scikit-learn, the
DecisionTreeClassifier with entropy criterion approximates C4.5 behavior.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt


def run_c45(X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, params: dict) -> dict:
    """
    Execute C4.5 (Decision Tree with entropy) classification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        params: Dictionary with parameters:
            - 'criterion': Split criterion ('entropy' for C4.5, 'gini') (default: 'entropy')
            - 'max_depth': Maximum tree depth (default: None = no limit)
            - 'min_samples_split': Minimum samples to split (default: 2)
            - 'min_samples_leaf': Minimum samples in leaf (default: 1)
        
    Returns:
        Dictionary with predictions, model, and feature importances
    """
    # C4.5 uses entropy (information gain)
    criterion = params.get("criterion", "entropy")
    max_depth = params.get("max_depth", None)
    min_samples_split = int(params.get("min_samples_split", 2))
    min_samples_leaf = int(params.get("min_samples_leaf", 1))
    
    # Create C4.5-like decision tree
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    
    # Train classifier
    dt.fit(X_train, y_train)
    
    # Predict
    predictions = dt.predict(X_test)
    
    # Get class probabilities
    try:
        probabilities = dt.predict_proba(X_test)
    except Exception:
        probabilities = None
    
    return {
        "predictions": predictions,
        "model": dt,
        "probabilities": probabilities,
        "feature_importances": dt.feature_importances_,
        "n_leaves": dt.get_n_leaves(),
        "max_depth_actual": dt.get_depth(),
        "classes": dt.classes_,
    }


def get_tree_rules(model: DecisionTreeClassifier, 
                   feature_names: list = None,
                   class_names: list = None) -> str:
    """
    Get human-readable decision rules from the tree.
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        
    Returns:
        String representation of decision rules
    """
    return export_text(
        model,
        feature_names=feature_names,
        class_names=class_names,
    )


def plot_decision_tree(model: DecisionTreeClassifier,
                       feature_names: list = None,
                       class_names: list = None,
                       figsize: tuple = (20, 10)) -> plt.Figure:
    """
    Create a visualization of the decision tree.
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=10,
    )
    
    ax.set_title("Arbre de Décision C4.5", fontsize=14, fontweight='bold')
    
    return fig


def get_feature_importance_ranking(model: DecisionTreeClassifier,
                                    feature_names: list = None) -> list:
    """
    Get feature importances ranked from most to least important.
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        
    Returns:
        List of tuples (feature_name, importance) sorted by importance
    """
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    # Sort by importance
    ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranking


def prune_tree(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               alpha_range: np.ndarray = None) -> dict:
    """
    Find optimal pruning parameter (ccp_alpha) using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        alpha_range: Range of alpha values to try
        
    Returns:
        Dictionary with best alpha and pruned tree
    """
    # Get cost-complexity pruning path
    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(X_train, y_train)
    
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Test different alpha values
    trees = []
    val_scores = []
    
    for ccp_alpha in ccp_alphas:
        dt_pruned = DecisionTreeClassifier(
            criterion="entropy",
            ccp_alpha=ccp_alpha,
            random_state=42
        )
        dt_pruned.fit(X_train, y_train)
        trees.append(dt_pruned)
        val_scores.append(dt_pruned.score(X_val, y_val))
    
    # Find best alpha
    best_idx = np.argmax(val_scores)
    
    return {
        "best_alpha": ccp_alphas[best_idx],
        "best_tree": trees[best_idx],
        "best_val_score": val_scores[best_idx],
        "all_alphas": ccp_alphas,
        "all_scores": val_scores,
    }
