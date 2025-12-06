"""
Tab components for classification results.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from classification import (
    run_classification,
    compute_all_metrics,
)
from classification.knn import evaluate_knn_k_range
from config.constants import COLOR_PALETTE


def render_confusion_matrix_plot(cm: np.ndarray, class_labels: list, title: str = "Confusion Matrix"):
    """
    Render an interactive confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_labels: List of class labels
        title: Plot title
    """
    # Convert to strings for display
    class_labels_str = [str(l) for l in class_labels]
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_labels_str,
        y=class_labels_str,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_summary(metrics: dict, title: str = "Performance Metrics"):
    """
    Render metrics summary in columns.
    
    Args:
        metrics: Dictionary of overall metrics
        title: Section title
    """
    st.markdown(f"### {title}")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics.get("accuracy", 0) * 100
        st.metric("Accuracy", f"{accuracy:.2f}%")
    
    with col2:
        precision = metrics.get("precision", metrics.get("precision_macro", 0)) * 100
        st.metric("Precision", f"{precision:.2f}%")
    
    with col3:
        recall = metrics.get("recall", metrics.get("recall_macro", 0)) * 100
        st.metric("Recall", f"{recall:.2f}%")
    
    with col4:
        f1 = metrics.get("f1", metrics.get("f1_macro", 0)) * 100
        st.metric("F-measure", f"{f1:.2f}%")


def render_per_class_metrics(per_class: dict):
    """
    Render per-class metrics in a table.
    
    Args:
        per_class: Dictionary of per-class metrics
    """
    st.markdown("### Per-class Metrics")
    
    # Build dataframe
    rows = []
    for class_label, metrics in per_class.items():
        rows.append({
            "Class": str(class_label),
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall": f"{metrics['recall']*100:.2f}%",
            "F1": f"{metrics['f1']*100:.2f}%",
            "Support": metrics["support"],
        })
    
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)


def render_knn_evaluation_tab(results: dict):
    """
    Render k-NN evaluation results for k=1 to 10.
    
    Args:
        results: Results from evaluate_knn_k_range
    """
    st.markdown("## k-NN Evaluation (k = 1 to 10)")
    st.markdown("*k-NN evaluation for different k values*")
    
    k_values = results["k_values"]
    
    # Create metrics dataframe
    df_results = pd.DataFrame({
        "k": k_values,
        "Accuracy (%)": [a * 100 for a in results["accuracy"]],
        "Precision (%)": [p * 100 for p in results["precision"]],
        "Recall (%)": [r * 100 for r in results["recall"]],
        "F-measure (%)": [f * 100 for f in results["f1"]],
    })
    
    # Display best k
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Best k (Accuracy)**: {results['best_k']} — {results['best_accuracy']*100:.2f}%")
    with col2:
        st.info(f"**Best k (F1)**: {results['best_k_f1']} — {results['best_f1']*100:.2f}%")
    
    # Display table
    st.markdown("### Results for each k value")
    st.dataframe(df_results.style.highlight_max(subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F-measure (%)"]),
                 hide_index=True, use_container_width=True)
    
    # Precision/Accuracy curve
    st.markdown("### Accuracy Curve (Accuracy vs k)")
    
    fig = go.Figure()
    
    # Accuracy curve
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[a * 100 for a in results["accuracy"]],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Precision curve
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[p * 100 for p in results["precision"]],
        mode='lines+markers',
        name='Precision',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # F1 curve
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[f * 100 for f in results["f1"]],
        mode='lines+markers',
        name='F-measure',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    ))
    
    # Mark best k
    fig.add_vline(
        x=results['best_k'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Best k = {results['best_k']}"
    )
    
    fig.update_layout(
        xaxis_title="Number of neighbors (k)",
        yaxis_title="Score (%)",
        title="k-NN Performance vs k",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices for each k
    st.markdown("### Confusion matrices by k value")
    
    # Select specific k to show
    selected_k = st.selectbox(
        "Select k to view confusion matrix",
        k_values,
        index=k_values.index(results['best_k'])
    )
    
    # Get confusion matrix for selected k
    cm = results["confusion_matrices"][k_values.index(selected_k)]
    class_labels = results["per_k_results"][selected_k]["metrics"]["class_labels"]
    
    render_confusion_matrix_plot(cm, class_labels, f"Confusion Matrix (k = {selected_k})")


def render_single_classifier_results(result: dict, algo_name: str):
    """
    Render results for a single classifier.
    
    Args:
        result: Classification result dictionary
        algo_name: Name of the algorithm
    """
    st.markdown(f"## Results: {algo_name}")
    
    # Overall metrics
    render_metrics_summary(result["metrics"])
    
    # Confusion matrix
    st.markdown("### Confusion Matrix")
    render_confusion_matrix_plot(
        result["confusion_matrix"],
        result["class_labels"],
        f"Confusion Matrix - {algo_name}"
    )
    
    # Per-class metrics
    render_per_class_metrics(result["per_class_metrics"])
    
    # Additional info based on algorithm
    if algo_name == "C4.5" and "feature_importances" in result:
        st.markdown("### Feature Importance")
        if result.get("feature_names"):
            importance_df = pd.DataFrame({
                "Feature": result["feature_names"],
                "Importance": result["feature_importances"]
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(importance_df, x="Feature", y="Importance",
                        title="Feature Importance (C4.5)")
            st.plotly_chart(fig, use_container_width=True)


def render_classifier_comparison(results: dict):
    """
    Render comparison of multiple classifiers.
    
    Args:
        results: Dictionary with results per algorithm
    """
    st.markdown("## Classifier Comparison")
    
    # Build comparison dataframe
    comparison_data = []
    
    for algo_name, result in results.items():
        if "error" in result:
            continue
        
        metrics = result["metrics"]
        comparison_data.append({
            "Algorithm": algo_name,
            "Accuracy (%)": metrics.get("accuracy", 0) * 100,
            "Precision (%)": metrics.get("precision", metrics.get("precision_macro", 0)) * 100,
            "Recall (%)": metrics.get("recall", metrics.get("recall_macro", 0)) * 100,
            "F-measure (%)": metrics.get("f1", metrics.get("f1_macro", 0)) * 100,
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Highlight best
    st.markdown("### Comparison Table")
    st.dataframe(
        df_comparison.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F-measure (%)"]
        ),
        hide_index=True,
        use_container_width=True
    )
    
    # Bar chart comparison
    st.markdown("### Comparative Visualization")
    
    metrics_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F-measure (%)"]
    
    fig = go.Figure()
    
    for metric in metrics_cols:
        fig.add_trace(go.Bar(
            name=metric.replace(" (%)", ""),
            x=df_comparison["Algorithm"],
            y=df_comparison[metric],
            text=[f"{v:.1f}%" for v in df_comparison[metric]],
            textposition="auto",
        ))
    
    fig.update_layout(
        barmode='group',
        title="Metric Comparison by Algorithm",
        xaxis_title="Algorithm",
        yaxis_title="Score (%)",
        legend_title="Metric",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find best algorithm
    if len(comparison_data) > 0:
        best_idx = df_comparison["Accuracy (%)"].idxmax()
        best_algo = df_comparison.loc[best_idx, "Algorithm"]
        best_accuracy = df_comparison.loc[best_idx, "Accuracy (%)"]
        st.success(f"**Best Classifier**: {best_algo} with {best_accuracy:.2f}% accuracy")


def render_classification_results_tab(df: pd.DataFrame, selected_features: list,
                                      target_col: str, split_config: dict,
                                      classifier_choice: str, classifier_params: dict):
    """
    Main tab for classification results.
    
    Args:
        df: Data DataFrame
        selected_features: List of feature column names
        target_col: Target column name
        split_config: Train/test split configuration
        classifier_choice: Selected classifier name
        classifier_params: Classifier parameters
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils.multiclass import type_of_target
    from utils.data_loader import validate_algorithm_compatibility
    
    # Validate target column exists
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' does not exist in dataset. Please select a valid column.")
        return
    
    # Validate selected features exist
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        st.error(f"❌ The following features do not exist in dataset: {missing_features}")
        return
    
    # Validate target is not continuous
    y_check = df[target_col].dropna().values
    target_type = type_of_target(y_check)
    
    if target_type == 'continuous' or target_type == 'continuous-multioutput':
        st.error(
            f"❌ Target variable '{target_col}' contains continuous values (detected type: {target_type}). "
            "Classification requires discrete classes (categorical). "
            "\n\n**Possible solutions:**"
            "\n- Select another column as target"
            "\n- Discretize the column (e.g. convert to categories)"
            "\n- Use a regression algorithm instead of classification"
        )
        return
    
    # Validate algorithm compatibility
    is_valid, validation_messages = validate_algorithm_compatibility(
        df, selected_features, classifier_choice, target_col
    )
    
    if not is_valid:
        for msg in validation_messages:
            st.error(f"❌ {msg}")
        return
    
    # Show warnings
    for msg in validation_messages:
        st.warning(f"⚠️ {msg}")
    
    # Special validation for Naive Bayes with negative values
    if classifier_choice == "Naive Bayes":
        X_check = df[selected_features]
        nb_type = classifier_params.get("type", "gaussian")
        if nb_type == "multinomial" and (X_check < 0).any().any():
            st.error("❌ Multinomial Naive Bayes cannot be used with negative values. "
                    "Please normalize your data (Min-Max) or use Gaussian type.")
            return
        if nb_type == "bernoulli":
            # Check if data looks binary
            unique_values = set()
            for col in selected_features:
                unique_values.update(X_check[col].unique())
            if not all(v in [0, 1] or pd.isna(v) for v in unique_values):
                st.warning("⚠️ Bernoulli Naive Bayes is optimized for binary data (0/1). "
                          "Your data does not appear to be binary.")
    
    # Prepare data
    X = df[selected_features].values
    y = df[target_col].values
    
    # Check class distribution for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()
    
    # Validate minimum samples per class for k-NN
    if classifier_choice == "k-NN":
        k_value = classifier_params.get("k", 5)
        if not classifier_params.get("evaluate_range", False):
            if min_class_count < k_value:
                st.warning(f"⚠️ The least represented class has only {min_class_count} samples, "
                          f"which is fewer than k={k_value}. Results potentially biased.")
    
    # Determine if stratification is possible
    can_stratify = split_config["stratify"] and min_class_count >= 2
    
    if split_config["stratify"] and not can_stratify:
        st.warning(f"⚠️ Stratification disabled: some classes have fewer than 2 samples (minimum: {min_class_count}).")
    
    stratify_val = y if can_stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify_val
        )
    except ValueError as e:
        st.error(f"Error during splitting: {e}")
        st.info("Try disabling stratification or using a dataset with more samples per class.")
        return
    
    st.markdown(f"**Training Set**: {len(X_train)} instances ({(1-split_config['test_size'])*100:.0f}%)")
    st.markdown(f"**Test Set**: {len(X_test)} instances ({split_config['test_size']*100:.0f}%)")
    st.markdown("---")
    
    # Special handling for k-NN with range evaluation
    if classifier_choice == "k-NN" and classifier_params.get("evaluate_range", False):
        knn_results = evaluate_knn_k_range(
            X_train, y_train, X_test, y_test,
            k_range=range(1, 11),
            metric=classifier_params.get("metric", "euclidean")
        )
        render_knn_evaluation_tab(knn_results)
    else:
        # Single classifier evaluation
        result = run_classification(
            classifier_choice,
            classifier_params,
            X_train, y_train,
            X_test, y_test
        )
        result["feature_names"] = selected_features
        render_single_classifier_results(result, classifier_choice)


def render_all_classifiers_comparison(df: pd.DataFrame, selected_features: list,
                                       target_col: str, split_config: dict):
    """
    Run and compare all classifiers.
    
    Args:
        df: Data DataFrame
        selected_features: List of feature column names
        target_col: Target column name
        split_config: Train/test split configuration
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils.multiclass import type_of_target
    from classification import compare_classifiers
    from utils.data_loader import validate_algorithm_compatibility
    
    # Validate target column
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' does not exist in dataset.")
        return
    
    # Validate target is not continuous
    y_check = df[target_col].dropna().values
    target_type = type_of_target(y_check)
    
    if target_type == 'continuous' or target_type == 'continuous-multioutput':
        st.error(
            f"❌ Target variable '{target_col}' contains continuous values (type: {target_type}). "
            "Classification requires discrete classes. "
            "Select a categorical column as target."
        )
        return
    
    # Check for missing values
    X_check = df[selected_features]
    if X_check.isnull().any().any():
        st.error("❌ Dataset contains missing values in selected features. "
                "Please perform preprocessing to handle missing values.")
        return
    
    # Prepare data
    X = df[selected_features].values
    y = df[target_col].values
    
    # Check number of classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        st.error("❌ Classification requires at least 2 distinct classes in target variable.")
        return
    
    min_class_count = class_counts.min()
    
    # Determine if stratification is possible
    can_stratify = split_config["stratify"] and min_class_count >= 2
    
    if split_config["stratify"] and not can_stratify:
        st.warning(f"⚠️ Stratification disabled: some classes have fewer than 2 samples (minimum: {min_class_count}).")
    
    stratify_val = y if can_stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify_val
        )
    except ValueError as e:
        st.error(f"Error during splitting: {e}")
        st.info("Try disabling stratification or using a dataset with more samples per class.")
        return
    
    st.markdown(f"**Training Set**: {len(X_train)} instances")
    st.markdown(f"**Test Set**: {len(X_test)} instances")
    st.markdown("---")
    
    # Run all classifiers with default params
    with st.spinner("Running all classifiers..."):
        results = compare_classifiers(X_train, y_train, X_test, y_test)
    
    render_classifier_comparison(results)
    
    # Individual results in expanders
    st.markdown("---")
    st.markdown("### Details by algorithm")
    
    for algo_name, result in results.items():
        if "error" in result:
            st.error(f"{algo_name}: {result['error']}")
            continue
        
        with st.expander(f"{algo_name}"):
            render_confusion_matrix_plot(
                result["confusion_matrix"],
                result["class_labels"],
                f"Confusion Matrix - {algo_name}"
            )
            render_per_class_metrics(result["per_class_metrics"])
