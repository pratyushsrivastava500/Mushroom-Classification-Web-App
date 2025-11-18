"""
Mushroom Classification Web App - Modular Version
A Streamlit application for binary classification of mushrooms
"""

import streamlit as st
from src.config import config
from src.data.data_loader import load_data, split_data
from src.components.sidebar import render_sidebar, render_data_checkbox
from src.components.main_content import render_main_content


def main():
    """Main application function"""
    # Set page title
    st.title(config.APP_TITLE)
    st.markdown(config.APP_SUBTITLE)
    
    # Load and split data
    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    
    # Render sidebar and get user selections
    params = render_sidebar()
    
    # Render main content with classification results
    render_main_content(params, x_train, x_test, y_train, y_test)
    
    # Render data display checkbox
    render_data_checkbox(df)


if __name__ == '__main__':
    main()


