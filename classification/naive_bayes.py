"""
Naive Bayes classifier implementation.

Naive Bayes is a probabilistic classifier based on Bayes' theorem with
strong (naive) independence assumptions between features.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def run_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, params: dict) -> dict:
    """
    Execute Naive Bayes classification.
    
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
        nb = GaussianNB(var_smoothing=var_smoothing)
        
    elif nb_type == "multinomial":
        # Multinomial NB requires non-negative features
        alpha = float(params.get("alpha", 1.0))
        nb = MultinomialNB(alpha=alpha)
        # Ensure non-negative values
        X_train = np.clip(X_train, 0, None)
        X_test = np.clip(X_test, 0, None)
        
    elif nb_type == "bernoulli":
        alpha = float(params.get("alpha", 1.0))
        binarize = params.get("binarize", 0.0)
        nb = BernoulliNB(alpha=alpha, binarize=binarize)
        
    else:
        # Default to Gaussian
        nb = GaussianNB()
    
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


def get_naive_bayes_params(model: GaussianNB) -> dict:
    """
    Get learned parameters from a trained Gaussian Naive Bayes model.
    
    Args:
        model: Trained GaussianNB model
        
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
