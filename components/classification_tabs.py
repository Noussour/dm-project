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


def render_confusion_matrix_plot(cm: np.ndarray, class_labels: list, title: str = "Matrice de Confusion"):
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
        labels=dict(x="Prédit", y="Réel", color="Nombre"),
        x=class_labels_str,
        y=class_labels_str,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Classe prédite",
        yaxis_title="Classe réelle",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_summary(metrics: dict, title: str = "Métriques de performance"):
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
        st.metric("Précision", f"{precision:.2f}%")
    
    with col3:
        recall = metrics.get("recall", metrics.get("recall_macro", 0)) * 100
        st.metric("Rappel", f"{recall:.2f}%")
    
    with col4:
        f1 = metrics.get("f1", metrics.get("f1_macro", 0)) * 100
        st.metric("F-mesure", f"{f1:.2f}%")


def render_per_class_metrics(per_class: dict):
    """
    Render per-class metrics in a table.
    
    Args:
        per_class: Dictionary of per-class metrics
    """
    st.markdown("### Métriques par classe")
    
    # Build dataframe
    rows = []
    for class_label, metrics in per_class.items():
        rows.append({
            "Classe": str(class_label),
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Précision": f"{metrics['precision']*100:.2f}%",
            "Rappel": f"{metrics['recall']*100:.2f}%",
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
    st.markdown("## Évaluation k-NN (k = 1 à 10)")
    st.markdown("*Évaluation de k-NN pour différentes valeurs de k*")
    
    k_values = results["k_values"]
    
    # Create metrics dataframe
    df_results = pd.DataFrame({
        "k": k_values,
        "Accuracy (%)": [a * 100 for a in results["accuracy"]],
        "Précision (%)": [p * 100 for p in results["precision"]],
        "Rappel (%)": [r * 100 for r in results["recall"]],
        "F-mesure (%)": [f * 100 for f in results["f1"]],
    })
    
    # Display best k
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Meilleur k (Accuracy)**: {results['best_k']} — {results['best_accuracy']*100:.2f}%")
    with col2:
        st.info(f"**Meilleur k (F1)**: {results['best_k_f1']} — {results['best_f1']*100:.2f}%")
    
    # Display table
    st.markdown("### Résultats pour chaque valeur de k")
    st.dataframe(df_results.style.highlight_max(subset=["Accuracy (%)", "Précision (%)", "Rappel (%)", "F-mesure (%)"]),
                 hide_index=True, use_container_width=True)
    
    # Precision/Accuracy curve
    st.markdown("### Courbe de précision (Accuracy vs k)")
    
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
        name='Précision',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    # F1 curve
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[f * 100 for f in results["f1"]],
        mode='lines+markers',
        name='F-mesure',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    ))
    
    # Mark best k
    fig.add_vline(
        x=results['best_k'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Meilleur k = {results['best_k']}"
    )
    
    fig.update_layout(
        xaxis_title="Nombre de voisins (k)",
        yaxis_title="Score (%)",
        title="Performance du k-NN en fonction de k",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices for each k
    st.markdown("### Matrices de confusion par valeur de k")
    
    # Select specific k to show
    selected_k = st.selectbox(
        "Sélectionnez k pour voir la matrice de confusion",
        k_values,
        index=k_values.index(results['best_k'])
    )
    
    # Get confusion matrix for selected k
    cm = results["confusion_matrices"][k_values.index(selected_k)]
    class_labels = results["per_k_results"][selected_k]["metrics"]["class_labels"]
    
    render_confusion_matrix_plot(cm, class_labels, f"Matrice de confusion (k = {selected_k})")


def render_single_classifier_results(result: dict, algo_name: str):
    """
    Render results for a single classifier.
    
    Args:
        result: Classification result dictionary
        algo_name: Name of the algorithm
    """
    st.markdown(f"## Résultats: {algo_name}")
    
    # Overall metrics
    render_metrics_summary(result["metrics"])
    
    # Confusion matrix
    st.markdown("### Matrice de Confusion")
    render_confusion_matrix_plot(
        result["confusion_matrix"],
        result["class_labels"],
        f"Matrice de Confusion - {algo_name}"
    )
    
    # Per-class metrics
    render_per_class_metrics(result["per_class_metrics"])
    
    # Additional info based on algorithm
    if algo_name == "C4.5" and "feature_importances" in result:
        st.markdown("### Importance des features")
        if result.get("feature_names"):
            importance_df = pd.DataFrame({
                "Feature": result["feature_names"],
                "Importance": result["feature_importances"]
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(importance_df, x="Feature", y="Importance",
                        title="Importance des features (C4.5)")
            st.plotly_chart(fig, use_container_width=True)


def render_classifier_comparison(results: dict):
    """
    Render comparison of multiple classifiers.
    
    Args:
        results: Dictionary with results per algorithm
    """
    st.markdown("## Comparaison des Classifieurs")
    
    # Build comparison dataframe
    comparison_data = []
    
    for algo_name, result in results.items():
        if "error" in result:
            continue
        
        metrics = result["metrics"]
        comparison_data.append({
            "Algorithme": algo_name,
            "Accuracy (%)": metrics.get("accuracy", 0) * 100,
            "Précision (%)": metrics.get("precision", metrics.get("precision_macro", 0)) * 100,
            "Rappel (%)": metrics.get("recall", metrics.get("recall_macro", 0)) * 100,
            "F-mesure (%)": metrics.get("f1", metrics.get("f1_macro", 0)) * 100,
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Highlight best
    st.markdown("### Tableau comparatif")
    st.dataframe(
        df_comparison.style.highlight_max(
            subset=["Accuracy (%)", "Précision (%)", "Rappel (%)", "F-mesure (%)"]
        ),
        hide_index=True,
        use_container_width=True
    )
    
    # Bar chart comparison
    st.markdown("### Visualisation comparative")
    
    metrics_cols = ["Accuracy (%)", "Précision (%)", "Rappel (%)", "F-mesure (%)"]
    
    fig = go.Figure()
    
    for metric in metrics_cols:
        fig.add_trace(go.Bar(
            name=metric.replace(" (%)", ""),
            x=df_comparison["Algorithme"],
            y=df_comparison[metric],
            text=[f"{v:.1f}%" for v in df_comparison[metric]],
            textposition="auto",
        ))
    
    fig.update_layout(
        barmode='group',
        title="Comparaison des métriques par algorithme",
        xaxis_title="Algorithme",
        yaxis_title="Score (%)",
        legend_title="Métrique",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find best algorithm
    if len(comparison_data) > 0:
        best_idx = df_comparison["Accuracy (%)"].idxmax()
        best_algo = df_comparison.loc[best_idx, "Algorithme"]
        best_accuracy = df_comparison.loc[best_idx, "Accuracy (%)"]
        st.success(f"**Meilleur classifieur**: {best_algo} avec {best_accuracy:.2f}% d'accuracy")


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
    
    # Validate target column exists
    if target_col not in df.columns:
        st.error(f"La colonne cible '{target_col}' n'existe pas dans le dataset. Veuillez sélectionner une colonne valide.")
        return
    
    # Validate selected features exist
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        st.error(f"Les features suivantes n'existent pas dans le dataset: {missing_features}")
        return
    
    # Prepare data
    X = df[selected_features].values
    y = df[target_col].values
    
    # Check class distribution for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()
    
    # Determine if stratification is possible
    can_stratify = split_config["stratify"] and min_class_count >= 2
    
    if split_config["stratify"] and not can_stratify:
        st.warning(f"Stratification désactivée: certaines classes ont moins de 2 échantillons (minimum: {min_class_count}).")
    
    stratify_val = y if can_stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify_val
        )
    except ValueError as e:
        st.error(f"Erreur lors du partitionnement: {e}")
        st.info("Essayez de désactiver la stratification ou d'utiliser un dataset avec plus d'échantillons par classe.")
        return
    
    st.markdown(f"**Ensemble d'apprentissage**: {len(X_train)} instances ({(1-split_config['test_size'])*100:.0f}%)")
    st.markdown(f"**Ensemble de test**: {len(X_test)} instances ({split_config['test_size']*100:.0f}%)")
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
    from classification import compare_classifiers
    
    # Prepare data
    X = df[selected_features].values
    y = df[target_col].values
    
    # Check class distribution for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = class_counts.min()
    
    # Determine if stratification is possible
    can_stratify = split_config["stratify"] and min_class_count >= 2
    
    if split_config["stratify"] and not can_stratify:
        st.warning(f"Stratification désactivée: certaines classes ont moins de 2 échantillons (minimum: {min_class_count}).")
    
    stratify_val = y if can_stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=stratify_val
        )
    except ValueError as e:
        st.error(f"Erreur lors du partitionnement: {e}")
        st.info("Essayez de désactiver la stratification ou d'utiliser un dataset avec plus d'échantillons par classe.")
        return
    
    st.markdown(f"**Ensemble d'apprentissage**: {len(X_train)} instances")
    st.markdown(f"**Ensemble de test**: {len(X_test)} instances")
    st.markdown("---")
    
    # Run all classifiers with default params
    with st.spinner("Exécution de tous les classifieurs..."):
        results = compare_classifiers(X_train, y_train, X_test, y_test)
    
    render_classifier_comparison(results)
    
    # Individual results in expanders
    st.markdown("---")
    st.markdown("### Détails par algorithme")
    
    for algo_name, result in results.items():
        if "error" in result:
            st.error(f"{algo_name}: {result['error']}")
            continue
        
        with st.expander(f"{algo_name}"):
            render_confusion_matrix_plot(
                result["confusion_matrix"],
                result["class_labels"],
                f"Matrice de Confusion - {algo_name}"
            )
            render_per_class_metrics(result["per_class_metrics"])
