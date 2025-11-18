"""
Main content component for displaying results and visualizations
"""

import streamlit as st
from src.models.model_trainer import ModelTrainer
from src.utils.visualization import plot_metrics, display_metrics
from src.config import config


def render_main_content(params, x_train, x_test, y_train, y_test):
    """
    Render the main content area with model results
    
    Args:
        params (dict): Dictionary containing classifier and parameters
        x_train: Training features
        x_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    classifier = params['classifier']
    
    if params['classify_button']:
        # Initialize the model trainer
        trainer = ModelTrainer(
            x_train, x_test, y_train, y_test, 
            config.CLASS_NAMES
        )
        
        # Train the model based on classifier selection
        if classifier == config.CLASSIFIER_SVM:
            st.subheader("Support Vector Machine (SVM) Results")
            results = trainer.train_svm(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma']
            )
            
        elif classifier == config.CLASSIFIER_LR:
            st.subheader("Logistic Regression Results")
            results = trainer.train_logistic_regression(
                C=params['C'],
                max_iter=params['max_iter']
            )
            
        elif classifier == config.CLASSIFIER_RF:
            st.subheader("Random Forest Results")
            results = trainer.train_random_forest(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                bootstrap=params['bootstrap']
            )
        
        # Display metrics
        display_metrics(results)
        
        # Plot selected visualizations
        if params['metrics']:
            plot_metrics(
                params['metrics'],
                results['model'],
                trainer.x_test,
                trainer.y_test,
                config.CLASS_NAMES
            )
