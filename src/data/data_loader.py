"""
Data loading and preprocessing module
"""

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config.config import DATA_PATH, DEFAULT_TEST_SIZE, RANDOM_STATE


@st.cache_data(persist=True)
def load_data(data_path=DATA_PATH):
    """
    Load and preprocess the mushroom dataset
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with label-encoded features
    """
    data = pd.read_csv(data_path)
    labelencoder = LabelEncoder()
    
    for col in data.columns:
        data[col] = labelencoder.fit_transform(data[col])
    
    return data


@st.cache_data(persist=True)
def split_data(df, test_size=DEFAULT_TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split the dataset into training and testing sets
    
    Args:
        df (pd.DataFrame): The dataset to split
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    y = df.type.copy()
    x = df.drop(columns=['type']).copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=test_size, 
        random_state=random_state
    )
    return x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
