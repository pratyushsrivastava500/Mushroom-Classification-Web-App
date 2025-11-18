"""
Visualization utilities for plotting model performance metrics
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from src.config.config import CLASS_LABELS


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    """
    Plot selected performance metrics
    
    Args:
        metrics_list (list): List of metrics to plot
        model: Trained model
        x_test: Test features
        y_test: Test labels
        class_names (list): List of class names
    """
    # Ensure data is in proper numpy format
    x_test = np.array(x_test)
    y_test = np.array(y_test).ravel()
    
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        # Let sklearn automatically determine labels from the data
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)
    
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)


def display_metrics(results):
    """
    Display model performance metrics
    
    Args:
        results (dict): Dictionary containing accuracy, precision, and recall
    """
    st.write("Accuracy: ", round(results['accuracy'], 2))
    st.write("Precision: ", round(results['precision'], 2))
    st.write("Recall: ", round(results['recall'], 2))
