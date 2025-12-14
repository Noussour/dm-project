"""
Naive Bayes classifier implementation.

Custom implementation from scratch.
Naive Bayes is a probabilistic classifier based on Bayes' theorem with
strong (naive) independence assumptions between features.
"""

import numpy as np
from typing import Optional


class GaussianNaiveBayes:
    """
    Custom Gaussian Naive Bayes Classifier.
    
    Assumes features follow a Gaussian (normal) distribution.
    
    Bayes Theorem: P(y|X) = P(X|y) * P(y) / P(X)
    
    For classification, we want argmax_y P(y|X), which is proportional to:
    argmax_y P(X|y) * P(y)
    
    With naive independence assumption:
    P(X|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)
    
    For Gaussian NB, P(xi|y) follows a normal distribution.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes.
        
        Args:
            var_smoothing: Portion of the largest variance to add for stability
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # Mean of each feature per class
        self.var_ = None    # Variance of each feature per class
        self._epsilon = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Naive Bayes according to X, y.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize arrays
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate priors, means, and variances for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[idx] = len(X_c) / len(X)
            self.theta_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0)
        
        # Add smoothing to variance for numerical stability
        self._epsilon = self.var_smoothing * self.var_.max()
        self.var_ += self._epsilon
        
        return self
    
    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood of samples for each class.
        
        Uses log form to avoid numerical underflow:
        log P(X|y) = sum_i log P(xi|y)
        
        For Gaussian: log P(x|y) = -0.5 * [log(2*pi*var) + (x-mean)^2/var]
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Log likelihood for each class (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            mean = self.theta_[idx]
            var = self.var_[idx]
            
            # Log of Gaussian probability density
            log_prob = -0.5 * (np.log(2 * np.pi * var) + ((X - mean) ** 2) / var)
            log_likelihood[:, idx] = log_prob.sum(axis=1)
        
        return log_likelihood
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log probability of each class for samples.
        
        log P(y|X) = log P(X|y) + log P(y) - log P(X)
        
        Args:
            X: Features
            
        Returns:
            Log probabilities (n_samples, n_classes)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        log_likelihood = self._log_likelihood(X)
        log_prior = np.log(self.class_prior_)
        
        # Unnormalized log posterior
        log_posterior = log_likelihood + log_prior
        
        # Normalize using log-sum-exp trick for numerical stability
        log_sum = np.log(np.sum(np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True)), axis=1, keepdims=True))
        log_proba = log_posterior - log_posterior.max(axis=1, keepdims=True) - log_sum
        
        return log_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probability of each class for samples.
        
        Args:
            X: Features
            
        Returns:
            Probabilities (n_samples, n_classes)
        """
        return np.exp(self.predict_log_proba(X))
    
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
        
        log_likelihood = self._log_likelihood(X)
        log_prior = np.log(self.class_prior_)
        
        # Posterior is proportional to likelihood * prior
        log_posterior = log_likelihood + log_prior
        
        # Return class with highest posterior probability
        return self.classes_[np.argmax(log_posterior, axis=1)]


class MultinomialNaiveBayes:
    """
    Custom Multinomial Naive Bayes Classifier.
    
    Suitable for discrete/count data (e.g., text classification).
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes.
        
        Args:
            alpha: Additive (Laplace) smoothing parameter
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Multinomial Naive Bayes.
        
        Args:
            X: Training features (non-negative counts)
            y: Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.class_prior_ = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[idx] = len(X_c) / len(X)
            feature_count[idx] = X_c.sum(axis=0) + self.alpha
        
        # Compute log probabilities
        smoothed_total = feature_count.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(feature_count / smoothed_total)
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute log probability of each class for samples."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        log_prior = np.log(self.class_prior_)
        log_likelihood = X @ self.feature_log_prob_.T
        log_posterior = log_likelihood + log_prior
        
        # Normalize
        log_sum = np.log(np.sum(np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True)), axis=1, keepdims=True))
        log_proba = log_posterior - log_posterior.max(axis=1, keepdims=True) - log_sum
        
        return log_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute probability of each class for samples."""
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        log_prior = np.log(self.class_prior_)
        log_likelihood = X @ self.feature_log_prob_.T
        log_posterior = log_likelihood + log_prior
        
        return self.classes_[np.argmax(log_posterior, axis=1)]


class BernoulliNaiveBayes:
    """
    Custom Bernoulli Naive Bayes Classifier.
    
    Suitable for binary/boolean features.
    """
    
    def __init__(self, alpha: float = 1.0, binarize: float = 0.0):
        """
        Initialize Bernoulli Naive Bayes.
        
        Args:
            alpha: Additive (Laplace) smoothing parameter
            binarize: Threshold for binarizing features
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_prior_ = None
        self.feature_prob_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Bernoulli Naive Bayes.
        
        Args:
            X: Training features
            y: Training labels
        """
        X = np.array(X)
        y = np.array(y)
        
        # Binarize features
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.class_prior_ = np.zeros(n_classes)
        self.feature_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = len(X_c)
            self.class_prior_[idx] = n_c / len(X)
            # Probability of feature = 1 given class
            self.feature_prob_[idx] = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)
        
        return self
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute log probability of each class for samples."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        log_prior = np.log(self.class_prior_)
        
        # P(X|y) = prod_i P(xi|y)^xi * (1-P(xi|y))^(1-xi)
        # log P(X|y) = sum_i [xi*log(P(xi|y)) + (1-xi)*log(1-P(xi|y))]
        log_prob = np.log(self.feature_prob_)
        log_1_prob = np.log(1 - self.feature_prob_)
        
        log_likelihood = X @ log_prob.T + (1 - X) @ log_1_prob.T
        log_posterior = log_likelihood + log_prior
        
        # Normalize
        log_sum = np.log(np.sum(np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True)), axis=1, keepdims=True))
        log_proba = log_posterior - log_posterior.max(axis=1, keepdims=True) - log_sum
        
        return log_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute probability of each class for samples."""
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        
        log_prior = np.log(self.class_prior_)
        log_prob = np.log(self.feature_prob_)
        log_1_prob = np.log(1 - self.feature_prob_)
        
        log_likelihood = X @ log_prob.T + (1 - X) @ log_1_prob.T
        log_posterior = log_likelihood + log_prior
        
        return self.classes_[np.argmax(log_posterior, axis=1)]


def run_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, params: dict) -> dict:
    """
    Execute Naive Bayes classification using custom implementation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        params: Dictionary with parameters:
            - 'type': Type of NB ('gaussian', 'multinomial', 'bernoulli') (default: 'gaussian')
            - 'var_smoothing': Smoothing parameter for GaussianNB (default: 1e-9)
            - 'alpha': Smoothing parameter for Multinomial/Bernoulli (default: 1.0)
        
    Returns:
        Dictionary with predictions, model, and probabilities
    """
    nb_type = params.get("type", "gaussian")
    
    if nb_type == "gaussian":
        var_smoothing = float(params.get("var_smoothing", 1e-9))
        nb = GaussianNaiveBayes(var_smoothing=var_smoothing)
        
    elif nb_type == "multinomial":
        # Multinomial NB requires non-negative features
        alpha = float(params.get("alpha", 1.0))
        nb = MultinomialNaiveBayes(alpha=alpha)
        # Ensure non-negative values
        X_train = np.clip(X_train, 0, None)
        X_test = np.clip(X_test, 0, None)
        
    elif nb_type == "bernoulli":
        alpha = float(params.get("alpha", 1.0))
        binarize = params.get("binarize", 0.0)
        nb = BernoulliNaiveBayes(alpha=alpha, binarize=binarize)
        
    else:
        # Default to Gaussian
        nb = GaussianNaiveBayes()
    
    # Train classifier
    nb.fit(X_train, y_train)
    
    # Predict
    predictions = nb.predict(X_test)
    
    # Get class probabilities
    probabilities = nb.predict_proba(X_test)
    
    return {
        "predictions": predictions,
        "model": nb,
        "probabilities": probabilities,
        "type": nb_type,
        "classes": nb.classes_,
    }


def get_naive_bayes_params(model: GaussianNaiveBayes) -> dict:
    """
    Get learned parameters from a trained Gaussian Naive Bayes model.
    
    Args:
        model: Trained GaussianNaiveBayes model
        
    Returns:
        Dictionary with class priors, means, and variances
    """
    return {
        "class_prior": model.class_prior_,  # P(class)
        "theta": model.theta_,  # Mean of features per class
        "var": model.var_,  # Variance of features per class
        "classes": model.classes_,
    }


def explain_prediction(model, X_sample: np.ndarray, feature_names: list = None) -> dict:
    """
    Explain a Naive Bayes prediction by showing probability contributions.
    
    Args:
        model: Trained Naive Bayes model
        X_sample: Single sample to explain (1D array)
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary with prediction explanation
    """
    X_sample = X_sample.reshape(1, -1)
    
    # Get prediction and probabilities
    prediction = model.predict(X_sample)[0]
    proba = model.predict_proba(X_sample)[0]
    
    # Log probabilities per class
    log_proba = model.predict_log_proba(X_sample)[0]
    
    explanation = {
        "prediction": prediction,
        "probabilities": dict(zip(model.classes_, proba)),
        "log_probabilities": dict(zip(model.classes_, log_proba)),
        "predicted_probability": float(max(proba)),
    }
    
    if feature_names is not None:
        explanation["features"] = feature_names
    
    return explanation
