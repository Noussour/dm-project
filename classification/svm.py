"""
Support Vector Machine (SVM) classifier implementation.

Custom implementation from scratch.
SVM finds the hyperplane that best separates classes with maximum margin.
Supports linear and non-linear kernels using SMO algorithm.
"""

import numpy as np
from typing import Optional


class StandardScaler:
    """
    Custom StandardScaler for feature normalization.
    
    Standardizes features by removing the mean and scaling to unit variance.
    z = (x - mean) / std
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray):
        """
        Compute mean and std to be used for later scaling.
        
        Args:
            X: Training data
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation."""
        return X * self.std_ + self.mean_


def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """Linear kernel: K(x1, x2) = x1 · x2"""
    return np.dot(x1, x2)


def polynomial_kernel(x1: np.ndarray, x2: np.ndarray, degree: int = 3, 
                      coef0: float = 1.0) -> float:
    """Polynomial kernel: K(x1, x2) = (x1 · x2 + coef0)^degree"""
    return (np.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 1.0) -> float:
    """RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))


def sigmoid_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 1.0,
                   coef0: float = 0.0) -> float:
    """Sigmoid kernel: K(x1, x2) = tanh(gamma * x1 · x2 + coef0)"""
    return np.tanh(gamma * np.dot(x1, x2) + coef0)


class BinarySVM:
    """
    Custom Binary SVM Classifier using simplified SMO algorithm.
    
    The SMO (Sequential Minimal Optimization) algorithm solves the 
    SVM quadratic programming problem by breaking it into smaller 
    sub-problems that can be solved analytically.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = "rbf",
                 gamma: float = 1.0, degree: int = 3,
                 tol: float = 1e-3, max_iter: int = 1000):
        """
        Initialize Binary SVM.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient for rbf/poly/sigmoid
            degree: Polynomial degree for 'poly' kernel
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
    
    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function between two samples."""
        if self.kernel == "linear":
            return linear_kernel(x1, x2)
        elif self.kernel == "rbf":
            return rbf_kernel(x1, x2, self.gamma)
        elif self.kernel == "poly":
            return polynomial_kernel(x1, x2, self.degree)
        elif self.kernel == "sigmoid":
            return sigmoid_kernel(x1, x2, self.gamma)
        else:
            return linear_kernel(x1, x2)
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix for training data."""
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k_val = self._compute_kernel(X[i], X[j])
                K[i, j] = k_val
                K[j, i] = k_val
        return K
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for samples.
        
        f(x) = sum_i(alpha_i * y_i * K(x_i, x)) + b
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        decision = np.zeros(n_samples)
        
        sv_indices = self.support_vector_indices_
        
        for i in range(n_samples):
            s = 0
            for j in sv_indices:
                s += self.alpha[j] * self.y_train[j] * \
                     self._compute_kernel(self.X_train[j], X[i])
            decision[i] = s + self.b
        
        return decision
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model using simplified SMO.
        
        Args:
            X: Training features
            y: Training labels (+1 or -1)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).astype(float)
        n_samples = X.shape[0]
        
        # Initialize alphas and bias
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(self.X_train)
        
        # SMO main loop
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                for i in range(n_samples):
                    num_changed += self._examine_example(i, K)
            else:
                # Loop over non-bound examples
                for i in range(n_samples):
                    if 0 < self.alpha[i] < self.C:
                        num_changed += self._examine_example(i, K)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
        
        # Find support vectors (alpha > 0)
        self.support_vector_indices_ = np.where(self.alpha > 1e-8)[0]
        self.support_vectors_ = self.X_train[self.support_vector_indices_]
        
        return self
    
    def _examine_example(self, i2: int, K: np.ndarray) -> int:
        """Examine example i2 for possible optimization."""
        y2 = self.y_train[i2]
        alpha2 = self.alpha[i2]
        E2 = self._compute_error(i2, K)
        r2 = E2 * y2
        
        # Check KKT conditions
        if (r2 < -self.tol and alpha2 < self.C) or \
           (r2 > self.tol and alpha2 > 0):
            
            # Try to find a good i1
            non_bound = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            
            if len(non_bound) > 1:
                # Use heuristic to choose i1
                i1 = self._select_j(i2, E2, K, non_bound)
                if self._take_step(i1, i2, K):
                    return 1
            
            # Try all non-bound examples
            for i1 in np.random.permutation(non_bound):
                if i1 != i2 and self._take_step(i1, i2, K):
                    return 1
            
            # Try all examples
            for i1 in np.random.permutation(len(self.y_train)):
                if i1 != i2 and self._take_step(i1, i2, K):
                    return 1
        
        return 0
    
    def _select_j(self, i: int, Ei: float, K: np.ndarray, 
                  candidates: np.ndarray) -> int:
        """Select j using maximum |Ei - Ej| heuristic."""
        max_diff = 0
        j = candidates[0]
        
        for c in candidates:
            if c != i:
                Ec = self._compute_error(c, K)
                diff = abs(Ei - Ec)
                if diff > max_diff:
                    max_diff = diff
                    j = c
        
        return j
    
    def _compute_error(self, i: int, K: np.ndarray) -> float:
        """Compute error for sample i: E_i = f(x_i) - y_i"""
        f_i = np.sum(self.alpha * self.y_train * K[:, i]) + self.b
        return f_i - self.y_train[i]
    
    def _take_step(self, i1: int, i2: int, K: np.ndarray) -> bool:
        """Take optimization step on samples i1 and i2."""
        if i1 == i2:
            return False
        
        alpha1_old = self.alpha[i1]
        alpha2_old = self.alpha[i2]
        y1 = self.y_train[i1]
        y2 = self.y_train[i2]
        
        E1 = self._compute_error(i1, K)
        E2 = self._compute_error(i2, K)
        
        s = y1 * y2
        
        # Compute L and H bounds
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L >= H:
            return False
        
        # Compute eta
        eta = 2 * K[i1, i2] - K[i1, i1] - K[i2, i2]
        
        if eta >= 0:
            return False
        
        # Compute new alpha2
        alpha2_new = alpha2_old - y2 * (E1 - E2) / eta
        
        # Clip alpha2
        if alpha2_new > H:
            alpha2_new = H
        elif alpha2_new < L:
            alpha2_new = L
        
        if abs(alpha2_new - alpha2_old) < 1e-5:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new)
        
        # Update bias
        b1 = self.b - E1 - y1 * (alpha1_new - alpha1_old) * K[i1, i1] - \
             y2 * (alpha2_new - alpha2_old) * K[i1, i2]
        b2 = self.b - E2 - y1 * (alpha1_new - alpha1_old) * K[i1, i2] - \
             y2 * (alpha2_new - alpha2_old) * K[i2, i2]
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update alphas
        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        
        return True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        decision = self._decision_function(X)
        return np.sign(decision)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision function values."""
        return self._decision_function(X)


class SVM:
    """
    Custom Multi-class SVM Classifier using One-vs-One strategy.
    """
    
    def __init__(self, C: float = 1.0, kernel: str = "rbf",
                 gamma: str = "scale", degree: int = 3,
                 probability: bool = True):
        """
        Initialize SVM.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            degree: Polynomial degree
            probability: Whether to enable probability estimates
        """
        self.C = C
        self.kernel = kernel
        self.gamma_param = gamma
        self.gamma = None
        self.degree = degree
        self.probability = probability
        
        self.classes_ = None
        self.classifiers_ = {}
        self.n_support_ = None
        self.support_vectors_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model using One-vs-One strategy.
        
        Args:
            X: Training features
            y: Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # Compute gamma
        if self.gamma_param == "scale":
            self.gamma = 1 / (n_features * X.var())
        elif self.gamma_param == "auto":
            self.gamma = 1 / n_features
        else:
            self.gamma = float(self.gamma_param)
        
        # Binary classification
        if len(self.classes_) == 2:
            # Map to -1, +1
            y_binary = np.where(y == self.classes_[0], -1, 1)
            
            clf = BinarySVM(C=self.C, kernel=self.kernel, 
                           gamma=self.gamma, degree=self.degree)
            clf.fit(X, y_binary)
            self.classifiers_[(0, 1)] = clf
            
            if clf.support_vectors_ is not None:
                self.support_vectors_ = clf.support_vectors_
                self.n_support_ = np.array([
                    np.sum(y[clf.support_vector_indices_] == self.classes_[0]),
                    np.sum(y[clf.support_vector_indices_] == self.classes_[1])
                ])
        else:
            # One-vs-One for multiclass
            all_support_vectors = []
            
            for i, c1 in enumerate(self.classes_):
                for j, c2 in enumerate(self.classes_):
                    if i < j:
                        # Get samples for these two classes
                        mask = (y == c1) | (y == c2)
                        X_pair = X[mask]
                        y_pair = y[mask]
                        
                        # Map to -1, +1
                        y_binary = np.where(y_pair == c1, -1, 1)
                        
                        clf = BinarySVM(C=self.C, kernel=self.kernel,
                                       gamma=self.gamma, degree=self.degree)
                        clf.fit(X_pair, y_binary)
                        self.classifiers_[(i, j)] = clf
                        
                        if clf.support_vectors_ is not None:
                            all_support_vectors.append(clf.support_vectors_)
            
            if all_support_vectors:
                self.support_vectors_ = np.vstack(all_support_vectors)
                self.n_support_ = np.zeros(len(self.classes_), dtype=int)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        
        if len(self.classes_) == 2:
            # Binary classification
            clf = self.classifiers_[(0, 1)]
            predictions_binary = clf.predict(X)
            return np.where(predictions_binary == -1, 
                          self.classes_[0], self.classes_[1])
        else:
            # Voting for multiclass
            votes = np.zeros((n_samples, len(self.classes_)))
            
            for (i, j), clf in self.classifiers_.items():
                predictions = clf.predict(X)
                for s in range(n_samples):
                    if predictions[s] == -1:
                        votes[s, i] += 1
                    else:
                        votes[s, j] += 1
            
            return self.classes_[np.argmax(votes, axis=1)]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision function values."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if len(self.classes_) == 2:
            clf = self.classifiers_[(0, 1)]
            return clf.decision_function(X)
        else:
            # Return max decision value for multiclass
            n_samples = X.shape[0]
            decisions = np.zeros((n_samples, len(self.classifiers_)))
            
            for idx, ((i, j), clf) in enumerate(self.classifiers_.items()):
                decisions[:, idx] = clf.decision_function(X)
            
            return decisions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate class probabilities using Platt scaling approximation.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary case: use sigmoid of decision function
            decision = self.decision_function(X)
            prob_pos = 1 / (1 + np.exp(-decision))
            proba = np.column_stack([1 - prob_pos, prob_pos])
        else:
            # Multiclass: use voting proportions
            votes = np.zeros((n_samples, n_classes))
            
            for (i, j), clf in self.classifiers_.items():
                predictions = clf.predict(X)
                for s in range(n_samples):
                    if predictions[s] == -1:
                        votes[s, i] += 1
                    else:
                        votes[s, j] += 1
            
            # Normalize votes to probabilities
            proba = votes / votes.sum(axis=1, keepdims=True)
        
        return proba


def run_svm(X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, params: dict) -> dict:
    """
    Execute SVM classification using custom implementation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        params: Dictionary with parameters:
            - 'kernel': Kernel type ('linear', 'rbf', 'poly', 'sigmoid') (default: 'rbf')
            - 'C': Regularization parameter (default: 1.0)
            - 'gamma': Kernel coefficient for rbf/poly/sigmoid (default: 'scale')
            - 'degree': Polynomial degree for 'poly' kernel (default: 3)
            - 'normalize': Whether to normalize features (default: True)
        
    Returns:
        Dictionary with predictions, model, and support vectors info
    """
    kernel = params.get("kernel", "rbf")
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    degree = int(params.get("degree", 3))
    normalize = params.get("normalize", True)
    
    # Normalize features if requested (recommended for SVM)
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Create SVM classifier
    svm = SVM(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
    )
    
    # Train classifier
    svm.fit(X_train_scaled, y_train)
    
    # Predict
    predictions = svm.predict(X_test_scaled)
    
    # Get class probabilities
    try:
        probabilities = svm.predict_proba(X_test_scaled)
    except Exception:
        probabilities = None
    
    return {
        "predictions": predictions,
        "model": svm,
        "scaler": scaler,
        "probabilities": probabilities,
        "support_vectors": svm.support_vectors_,
        "n_support": svm.n_support_,
        "classes": svm.classes_,
        "kernel": kernel,
    }


def tune_svm_parameters(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None,
                        use_cv: bool = True) -> dict:
    """
    Find optimal SVM parameters using grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        use_cv: Whether to use cross-validation
        
    Returns:
        Dictionary with best parameters and results
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # Parameter grid (simplified)
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['linear', 'rbf'],
    }
    
    best_params = None
    best_score = -1
    best_model = None
    
    # Simple grid search
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            for kernel in param_grid['kernel']:
                svm = SVM(C=C, gamma=gamma, kernel=kernel)
                svm.fit(X_train_scaled, y_train)
                
                if X_val is not None:
                    predictions = svm.predict(X_val_scaled)
                    score = np.mean(predictions == y_val)
                else:
                    # Use training accuracy if no validation set
                    predictions = svm.predict(X_train_scaled)
                    score = np.mean(predictions == y_train)
                
                if score > best_score:
                    best_score = score
                    best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
                    best_model = svm
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_estimator": best_model,
        "scaler": scaler,
    }


def get_decision_boundary_data(svm_model, scaler, X: np.ndarray, 
                                feature_indices: tuple = (0, 1),
                                resolution: int = 100) -> dict:
    """
    Get data for plotting decision boundaries (2D only).
    
    Args:
        svm_model: Trained SVM model
        scaler: Feature scaler used during training
        X: Feature array
        feature_indices: Tuple of two feature indices to plot
        resolution: Grid resolution
        
    Returns:
        Dictionary with meshgrid and decision values
    """
    i, j = feature_indices
    
    # Get feature ranges
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    
    # Create meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Create feature matrix for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Add zeros for other features if needed
    n_features = X.shape[1]
    if n_features > 2:
        full_grid = np.zeros((grid_points.shape[0], n_features))
        full_grid[:, i] = grid_points[:, 0]
        full_grid[:, j] = grid_points[:, 1]
        grid_points = full_grid
    
    # Scale and predict
    if scaler is not None:
        grid_points_scaled = scaler.transform(grid_points)
    else:
        grid_points_scaled = grid_points
    
    Z = svm_model.predict(grid_points_scaled)
    
    # Handle non-numeric class labels
    unique_classes = svm_model.classes_
    Z_numeric = np.zeros_like(Z, dtype=float)
    for idx, cls in enumerate(unique_classes):
        Z_numeric[Z == cls] = idx
    
    Z_numeric = Z_numeric.reshape(xx.shape)
    
    return {
        "xx": xx,
        "yy": yy,
        "Z": Z_numeric,
        "feature_indices": feature_indices,
    }


def explain_svm_prediction(svm_model, scaler, X_sample: np.ndarray, 
                            y_sample: int = None) -> dict:
    """
    Explain an SVM prediction by showing distance to decision boundary.
    
    Args:
        svm_model: Trained SVM model
        scaler: Feature scaler
        X_sample: Single sample to explain
        y_sample: True label (optional)
        
    Returns:
        Dictionary with prediction explanation
    """
    X_sample = X_sample.reshape(1, -1)
    
    if scaler is not None:
        X_scaled = scaler.transform(X_sample)
    else:
        X_scaled = X_sample
    
    # Prediction
    prediction = svm_model.predict(X_scaled)[0]
    
    # Distance to decision boundary
    decision_values = svm_model.decision_function(X_scaled)[0]
    
    # Probability (if available)
    try:
        proba = svm_model.predict_proba(X_scaled)[0]
    except Exception:
        proba = None
    
    explanation = {
        "prediction": prediction,
        "decision_function": decision_values,
        "confidence": abs(decision_values) if not isinstance(decision_values, np.ndarray) else np.max(np.abs(decision_values)),
    }
    
    if proba is not None:
        explanation["probabilities"] = dict(zip(svm_model.classes_, proba))
    
    if y_sample is not None:
        explanation["true_label"] = y_sample
        explanation["correct"] = prediction == y_sample
    
    return explanation
