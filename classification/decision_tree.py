"""
C4.5 Decision Tree classifier implementation.

Custom implementation from scratch.
C4.5 is a decision tree algorithm developed by Ross Quinlan that uses
information gain ratio for feature selection.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt


class DecisionTreeNode:
    """
    Node in the decision tree.
    
    Attributes:
        feature_index: Index of feature to split on (None for leaf)
        threshold: Threshold value for split (None for leaf)
        left: Left child node (values <= threshold)
        right: Right child node (values > threshold)
        value: Class label for leaf node
        class_distribution: Distribution of classes in this node
        n_samples: Number of samples in this node
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.class_distribution = None
        self.n_samples = 0
        self.entropy = 0.0


class C45DecisionTree:
    """
    Custom C4.5 Decision Tree Classifier.
    
    Uses entropy and information gain ratio for splits.
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = "entropy"):
        """
        Initialize C4.5 Decision Tree.
        
        Args:
            max_depth: Maximum depth of tree (None for unlimited)
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            criterion: Split criterion ('entropy' for C4.5, 'gini' for CART)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a label array.
        
        H(S) = -sum(p_i * log2(p_i))
        
        Args:
            y: Array of class labels
            
        Returns:
            Entropy value
        """
        if len(y) == 0:
            return 0.0
        
        # Calculate class probabilities
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        
        # Calculate entropy (avoid log(0))
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity of a label array.
        
        Gini(S) = 1 - sum(p_i^2)
        
        Args:
            y: Array of class labels
            
        Returns:
            Gini impurity value
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        
        return 1 - np.sum(probs ** 2)
    
    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == "entropy":
            return self._entropy(y)
        else:
            return self._gini(y)
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, 
                          y_right: np.ndarray) -> float:
        """
        Calculate information gain from a split.
        
        IG = H(parent) - weighted average H(children)
        
        Args:
            y: Parent labels
            y_left: Left child labels
            y_right: Right child labels
            
        Returns:
            Information gain value
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_impurity = self._impurity(y)
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                         (n_right / n) * self._impurity(y_right)
        
        return parent_impurity - child_impurity
    
    def _gain_ratio(self, y: np.ndarray, y_left: np.ndarray, 
                    y_right: np.ndarray) -> float:
        """
        Calculate gain ratio (C4.5 improvement over ID3).
        
        GainRatio = InformationGain / SplitInfo
        
        This helps avoid bias towards features with many values.
        
        Args:
            y: Parent labels
            y_left: Left child labels
            y_right: Right child labels
            
        Returns:
            Gain ratio value
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        info_gain = self._information_gain(y, y_left, y_right)
        
        # Split info (intrinsic value)
        p_left = n_left / n
        p_right = n_right / n
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)
        
        if split_info == 0:
            return 0.0
        
        return info_gain / split_info
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find the best feature and threshold to split on.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try each unique value as a threshold
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Check minimum samples constraints
                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate gain ratio (C4.5)
                gain = self._gain_ratio(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """
        Recursively build the decision tree.
        
        Args:
            X: Features
            y: Labels
            depth: Current depth
            
        Returns:
            Root node of subtree
        """
        node = DecisionTreeNode()
        node.n_samples = len(y)
        node.entropy = self._entropy(y)
        
        # Calculate class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        node.class_distribution = dict(zip(unique_classes, counts))
        
        # Check stopping conditions
        n_samples, n_features = X.shape
        n_labels = len(unique_classes)
        
        # If pure node or max depth reached or not enough samples
        if n_labels == 1 or \
           (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            node.value = unique_classes[np.argmax(counts)]
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found
        if best_feature is None:
            node.value = unique_classes[np.argmax(counts)]
            return node
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        # Recursively build children
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build the decision tree from training data.
        
        Args:
            X: Training features
            y: Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        
        # Compute feature importances
        self._compute_feature_importances(X, y)
        
        return self
    
    def _compute_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Compute feature importances based on information gain at each split."""
        self.feature_importances_ = np.zeros(self.n_features_)
        total_samples = len(y)
        
        def _traverse(node, n_samples):
            if node is None or node.value is not None:
                return
            
            # Weight by samples at this node
            weight = n_samples / total_samples
            
            # Estimate gain at this node
            left_fraction = node.left.n_samples / n_samples if node.left else 0
            right_fraction = node.right.n_samples / n_samples if node.right else 0
            
            gain = node.entropy
            if node.left:
                gain -= left_fraction * node.left.entropy
            if node.right:
                gain -= right_fraction * node.right.entropy
            
            self.feature_importances_[node.feature_index] += weight * gain
            
            if node.left:
                _traverse(node.left, node.left.n_samples)
            if node.right:
                _traverse(node.right, node.right.n_samples)
        
        _traverse(self.root, total_samples)
        
        # Normalize
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
    
    def _predict_single(self, x: np.ndarray, node: DecisionTreeNode):
        """Predict class for a single sample."""
        # Leaf node
        if node.value is not None:
            return node.value
        
        # Internal node
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = [self._predict_single(x, self.root) for x in X]
        return np.array(predictions)
    
    def _predict_proba_single(self, x: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        """Get class probabilities for a single sample."""
        # Traverse to leaf
        if node.value is not None:
            # Return class distribution as probabilities
            proba = np.zeros(len(self.classes_))
            for cls, count in node.class_distribution.items():
                idx = np.where(self.classes_ == cls)[0][0]
                proba[idx] = count / node.n_samples
            return proba
        
        # Internal node
        if x[node.feature_index] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        probabilities = [self._predict_proba_single(x, self.root) for x in X]
        return np.array(probabilities)
    
    def get_depth(self) -> int:
        """Get the actual depth of the tree."""
        def _depth(node):
            if node is None or node.value is not None:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)
    
    def get_n_leaves(self) -> int:
        """Get the number of leaves in the tree."""
        def _count_leaves(node):
            if node is None:
                return 0
            if node.value is not None:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(self.root)


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
    
    # Create C4.5 decision tree
    dt = C45DecisionTree(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
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


def get_tree_rules(model: C45DecisionTree, 
                   feature_names: list = None,
                   class_names: list = None) -> str:
    """
    Get human-readable decision rules from the tree.
    
    Args:
        model: Trained C45DecisionTree
        feature_names: List of feature names
        class_names: List of class names
        
    Returns:
        String representation of decision rules
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(model.n_features_)]
    
    rules = []
    
    def _generate_rules(node, prefix=""):
        if node.value is not None:
            # Leaf node
            class_name = class_names[np.where(model.classes_ == node.value)[0][0]] \
                         if class_names else str(node.value)
            rules.append(f"{prefix}class: {class_name}")
            return
        
        feature_name = feature_names[node.feature_index]
        
        # Left branch
        rules.append(f"{prefix}|--- {feature_name} <= {node.threshold:.2f}")
        _generate_rules(node.left, prefix + "|   ")
        
        # Right branch
        rules.append(f"{prefix}|--- {feature_name} > {node.threshold:.2f}")
        _generate_rules(node.right, prefix + "|   ")
    
    _generate_rules(model.root)
    return "\n".join(rules)


def plot_decision_tree(model: C45DecisionTree,
                       feature_names: list = None,
                       class_names: list = None,
                       figsize: tuple = (20, 10)) -> plt.Figure:
    """
    Create a visualization of the decision tree.
    
    Args:
        model: Trained C45DecisionTree
        feature_names: List of feature names
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if feature_names is None:
        feature_names = [f"X[{i}]" for i in range(model.n_features_)]
    
    # Simple text-based tree visualization
    tree_text = get_tree_rules(model, feature_names, class_names)
    ax.text(0.1, 0.9, tree_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            family='monospace')
    ax.axis('off')
    ax.set_title("C4.5 Decision Tree", fontsize=14, fontweight='bold')
    
    return fig


def get_feature_importance_ranking(model: C45DecisionTree,
                                    feature_names: list = None) -> list:
    """
    Get feature importances ranked from most to least important.
    
    Args:
        model: Trained C45DecisionTree
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
    Find optimal tree depth using validation set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        alpha_range: Range of max_depth values to try
        
    Returns:
        Dictionary with best parameters and pruned tree
    """
    if alpha_range is None:
        alpha_range = list(range(1, 21))  # Test depths 1-20
    
    # Test different max_depth values
    trees = []
    val_scores = []
    
    for max_depth in alpha_range:
        dt = C45DecisionTree(max_depth=max_depth)
        dt.fit(X_train, y_train)
        
        # Calculate validation accuracy
        predictions = dt.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        
        trees.append(dt)
        val_scores.append(accuracy)
    
    # Find best depth
    best_idx = np.argmax(val_scores)
    
    return {
        "best_alpha": alpha_range[best_idx],
        "best_tree": trees[best_idx],
        "best_val_score": val_scores[best_idx],
        "all_alphas": alpha_range,
        "all_scores": val_scores,
    }
