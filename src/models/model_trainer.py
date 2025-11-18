"""
Model training and evaluation module
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, x_train, x_test, y_train, y_test, class_names):
        """
        Initialize the ModelTrainer
        
        Args:
            x_train: Training features
            x_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            class_names: List of class names
        """
        # Create writable copies and ensure proper numpy array types
        self.x_train = np.array(x_train.copy() if hasattr(x_train, 'copy') else x_train)
        self.x_test = np.array(x_test.copy() if hasattr(x_test, 'copy') else x_test)
        self.y_train = np.array(y_train.copy() if hasattr(y_train, 'copy') else y_train).ravel()
        self.y_test = np.array(y_test.copy() if hasattr(y_test, 'copy') else y_test).ravel()
        self.class_names = class_names
        self.model = None
        self.y_pred = None
    
    def train_svm(self, C=1.0, kernel='rbf', gamma='scale'):
        """
        Train Support Vector Machine model
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('rbf' or 'linear')
            gamma (str): Kernel coefficient
            
        Returns:
            dict: Model performance metrics
        """
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)
        self.model.fit(self.x_train, self.y_train)
        return self._evaluate_model()
    
    def train_logistic_regression(self, C=1.0, max_iter=100):
        """
        Train Logistic Regression model
        
        Args:
            C (float): Regularization parameter
            max_iter (int): Maximum number of iterations
            
        Returns:
            dict: Model performance metrics
        """
        self.model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
        self.model.fit(self.x_train, self.y_train)
        return self._evaluate_model()
    
    def train_random_forest(self, n_estimators=100, max_depth=5, bootstrap=True):
        """
        Train Random Forest model
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the tree
            bootstrap (bool): Whether to use bootstrap samples
            
        Returns:
            dict: Model performance metrics
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            bootstrap=bootstrap,
            n_jobs=-1
        )
        self.model.fit(self.x_train, self.y_train)
        return self._evaluate_model()
    
    def _evaluate_model(self):
        """
        Evaluate the trained model
        
        Returns:
            dict: Dictionary containing accuracy, precision, and recall
        """
        accuracy = self.model.score(self.x_test, self.y_test)
        self.y_pred = self.model.predict(self.x_test)
        
        # Use average='binary' for binary classification
        precision = precision_score(
            self.y_test, 
            self.y_pred, 
            average='binary'
        )
        recall = recall_score(
            self.y_test, 
            self.y_pred, 
            average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'model': self.model,
            'y_pred': self.y_pred
        }
    
    def get_model(self):
        """Get the trained model"""
        return self.model
    
    def get_predictions(self):
        """Get the model predictions"""
        return self.y_pred
