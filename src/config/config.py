"""
Configuration settings for the Mushroom Classification Web App
"""

# Application settings
APP_TITLE = "Binary Classification Web App"
APP_SUBTITLE = "Are your mushrooms edible or poisonous? 🍄"

# Data settings
DATA_PATH = "data/mushrooms.csv"
CLASS_NAMES = [0, 1]  # Numeric labels after encoding (0=edible, 1=poisonous)
CLASS_LABELS = ['edible', 'poisonous']  # Display names for visualization

# Model hyperparameters
DEFAULT_TEST_SIZE = 0.3
RANDOM_STATE = 0

# SVM default parameters
SVM_DEFAULT_C = 1.0
SVM_DEFAULT_KERNEL = "rbf"
SVM_DEFAULT_GAMMA = "scale"
SVM_C_RANGE = (0.01, 10.0)
SVM_C_STEP = 0.01

# Logistic Regression default parameters
LR_DEFAULT_C = 1.0
LR_DEFAULT_MAX_ITER = 100
LR_C_RANGE = (0.01, 10.0)
LR_C_STEP = 0.01
LR_MAX_ITER_RANGE = (100, 500)
LR_PENALTY = 'l2'

# Random Forest default parameters
RF_DEFAULT_N_ESTIMATORS = 100
RF_DEFAULT_MAX_DEPTH = 5
RF_DEFAULT_BOOTSTRAP = True
RF_N_ESTIMATORS_RANGE = (100, 5000)
RF_N_ESTIMATORS_STEP = 10
RF_MAX_DEPTH_RANGE = (1, 20)
RF_N_JOBS = -1

# Available metrics for visualization
AVAILABLE_METRICS = [
    'Confusion Matrix',
    'ROC Curve',
    'Precision-Recall Curve'
]

# Classifier names
CLASSIFIER_SVM = "Support Vector Machine (SVM)"
CLASSIFIER_LR = "Logistic Regression"
CLASSIFIER_RF = "Random Forest"
AVAILABLE_CLASSIFIERS = [CLASSIFIER_SVM, CLASSIFIER_LR, CLASSIFIER_RF]
