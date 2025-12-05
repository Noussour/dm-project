"""
Support Vector Machine (SVM) classifier implementation.

SVM finds the hyperplane that best separates classes with maximum margin.
Supports linear and non-linear kernels.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def run_svm(X_train: np.ndarray, y_train: np.ndarray, 
            X_test: np.ndarray, params: dict) -> dict:
    """
    Execute SVM classification.
    
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
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        random_state=42,
        probability=True,  # Enable probability estimates
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
        "n_support": svm.n_support_,  # Number of support vectors per class
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
        use_cv: Whether to use cross-validation (if no validation set)
        
    Returns:
        Dictionary with best parameters and results
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
    }
    
    # Grid search with cross-validation
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)
    
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
        "cv_results": grid_search.cv_results_,
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
    Z = Z.reshape(xx.shape)
    
    return {
        "xx": xx,
        "yy": yy,
        "Z": Z,
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
