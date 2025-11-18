"""
Sidebar component for classifier selection and hyperparameter tuning
"""

import streamlit as st
from src.config import config


def render_sidebar():
    """
    Render the sidebar with classifier selection and hyperparameters
    
    Returns:
        dict: Dictionary containing classifier name and parameters
    """
    st.sidebar.title(config.APP_TITLE)
    st.sidebar.markdown(config.APP_SUBTITLE)
    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", 
        config.AVAILABLE_CLASSIFIERS
    )
    
    params = {}
    params['classifier'] = classifier
    
    if classifier == config.CLASSIFIER_SVM:
        params.update(_get_svm_params())
    elif classifier == config.CLASSIFIER_LR:
        params.update(_get_lr_params())
    elif classifier == config.CLASSIFIER_RF:
        params.update(_get_rf_params())
    
    # Metrics selection
    params['metrics'] = st.sidebar.multiselect(
        "What metrics to plot?", 
        config.AVAILABLE_METRICS
    )
    
    # Classify button
    params['classify_button'] = st.sidebar.button("Classify", key='classify')
    
    return params


def _get_svm_params():
    """Get SVM hyperparameters from sidebar"""
    st.sidebar.subheader("Model Hyperparameters")
    
    C = st.sidebar.number_input(
        "C (Regularization parameter)", 
        config.SVM_C_RANGE[0], 
        config.SVM_C_RANGE[1], 
        step=config.SVM_C_STEP, 
        key='C_SVM'
    )
    kernel = st.sidebar.radio(
        "Kernel", 
        ("rbf", "linear"), 
        key='kernel'
    )
    gamma = st.sidebar.radio(
        "Gamma (Kernel Coefficient)", 
        ("scale", "auto"), 
        key='gamma'
    )
    
    return {'C': C, 'kernel': kernel, 'gamma': gamma}


def _get_lr_params():
    """Get Logistic Regression hyperparameters from sidebar"""
    st.sidebar.subheader("Model Hyperparameters")
    
    C = st.sidebar.number_input(
        "C (Regularization parameter)", 
        config.LR_C_RANGE[0], 
        config.LR_C_RANGE[1], 
        step=config.LR_C_STEP, 
        key='C_LR'
    )
    max_iter = st.sidebar.slider(
        "Maximum number of iterations", 
        config.LR_MAX_ITER_RANGE[0], 
        config.LR_MAX_ITER_RANGE[1], 
        key='max_iter'
    )
    
    return {'C': C, 'max_iter': max_iter}


def _get_rf_params():
    """Get Random Forest hyperparameters from sidebar"""
    st.sidebar.subheader("Model Hyperparameters")
    
    n_estimators = st.sidebar.number_input(
        "The number of trees in the forest", 
        config.RF_N_ESTIMATORS_RANGE[0], 
        config.RF_N_ESTIMATORS_RANGE[1], 
        step=config.RF_N_ESTIMATORS_STEP, 
        key='n_estimators'
    )
    max_depth = st.sidebar.number_input(
        "The maximum depth of the tree", 
        config.RF_MAX_DEPTH_RANGE[0], 
        config.RF_MAX_DEPTH_RANGE[1], 
        step=1, 
        key='max_depth'
    )
    bootstrap = st.sidebar.radio(
        "Bootstrap samples when building trees", 
        ('True', 'False'), 
        key='bootstrap'
    )
    
    # Convert bootstrap string to boolean
    bootstrap = True if bootstrap == 'True' else False
    
    return {'n_estimators': n_estimators, 'max_depth': max_depth, 'bootstrap': bootstrap}


def render_data_checkbox(df):
    """
    Render checkbox to show/hide raw data
    
    Args:
        df: The dataframe to display
    """
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
        st.markdown(
            "This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) "
            "includes descriptions of hypothetical samples corresponding to 23 species "
            "of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). "
            "Each species is identified as definitely edible, definitely poisonous, "
            "or of unknown edibility and not recommended. This latter class was "
            "combined with the poisonous one."
        )
