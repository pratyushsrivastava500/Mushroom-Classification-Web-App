#  Mushroom Classification Web App

An interactive machine learning web application built with **Streamlit** that classifies mushrooms as **edible** or **poisonous** using various ML algorithms. Users can experiment with different classification models and visualize their performance metrics in real-time.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance Metrics](#model-performance-metrics)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project is a **binary classification web application** that helps identify whether mushrooms are safe to eat or poisonous. The app provides an interactive interface where users can:
- Choose from multiple machine learning algorithms
- Tune hyperparameters in real-time
- Visualize model performance with various metrics
- Explore the mushroom dataset

The application is designed for educational purposes and demonstrates the practical implementation of machine learning classification algorithms.

### Modular Architecture

This project follows a **modular architecture** for better code organization, maintainability, and scalability:

- **Separation of Concerns**: Each module has a specific responsibility (data handling, model training, visualization, UI components)
- **Reusability**: Components can be easily reused or extended
- **Testability**: Individual modules can be tested independently
- **Maintainability**: Changes to one module have minimal impact on others
- **Scalability**: Easy to add new models, metrics, or features

The modular structure makes it easy to:
- Add new machine learning algorithms
- Extend visualization capabilities
- Modify UI components without affecting business logic
- Update configuration without touching code

##  Features

- **Multiple ML Algorithms**: Choose from 3 different classifiers
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  
- **Interactive Hyperparameter Tuning**: Adjust model parameters on the fly
  - Regularization parameters
  - Kernel selection
  - Number of estimators
  - Maximum depth
  - And more!

- **Real-time Performance Metrics**:
  - Accuracy Score
  - Precision
  - Recall
  
- **Visual Analytics**:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve

- **Data Exploration**: View and explore the raw mushroom dataset

##  Dataset

The application uses the **Mushroom Classification Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom).

### Dataset Details:
- **Source**: UCI ML Repository
- **Instances**: 8,124 mushroom samples
- **Features**: 23 categorical attributes
- **Classes**: 2 (Edible, Poisonous)
- **Species**: 23 species of gilled mushrooms from Agaricus and Lepiota Family

### Key Features:
- `type`: Edible (e) or Poisonous (p)
- `cap_shape`: Bell, conical, convex, flat, etc.
- `cap_surface`: Fibrous, grooves, scaly, smooth
- `cap_color`: Brown, buff, cinnamon, gray, green, pink, purple, red, white, yellow
- `bruises`: Bruises or not
- `odor`: Almond, anise, creosote, fishy, foul, musty, none, pungent, spicy
- `gill_attachment`, `gill_spacing`, `gill_size`, `gill_color`
- `stalk_shape`, `stalk_root`, `stalk_surface_above_ring`, `stalk_surface_below_ring`
- `stalk_color_above_ring`, `stalk_color_below_ring`
- `veil_type`, `veil_color`
- `ring_number`, `ring_type`
- `spore_print_color`
- `population`
- `habitat`: Grasses, leaves, meadows, paths, urban, waste, woods

##  Machine Learning Models

### 1. Support Vector Machine (SVM)
- **Tunable Parameters**:
  - C (Regularization parameter): 0.01 to 10.0
  - Kernel: RBF or Linear
  - Gamma: Scale or Auto
- **Best For**: High-dimensional data, clear margin of separation

### 2. Logistic Regression
- **Tunable Parameters**:
  - C (Regularization parameter): 0.01 to 10.0
  - Max Iterations: 100 to 500
  - Penalty: L2
- **Best For**: Binary classification, interpretable results

### 3. Random Forest
- **Tunable Parameters**:
  - Number of estimators: 100 to 5000
  - Max depth: 1 to 20
  - Bootstrap: True or False
- **Best For**: Handling non-linear relationships, feature importance

##  Project Structure

```
Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not/
│
├── data/                         # Data directory
│   └── mushrooms.csv            # Mushroom classification dataset
│
├── src/                          # Source code directory
│   ├── __init__.py              # Package initialization
│   ├── config/                   # Configuration module
│   │   ├── __init__.py
│   │   └── config.py            # Application settings and constants
│   ├── data/                     # Data handling module
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data loading and preprocessing
│   ├── models/                   # Model training module
│   │   ├── __init__.py
│   │   └── model_trainer.py     # Model training and evaluation
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── visualization.py     # Plotting and visualization
│   └── components/               # UI components
│       ├── __init__.py
│       ├── sidebar.py           # Sidebar component
│       └── main_content.py      # Main content component
│
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                     # Project documentation
```

### Module Description

#### `data/`
Contains the dataset:
- `mushrooms.csv`: The mushroom classification dataset from UCI ML Repository

#### `src/config/`
Contains configuration settings and constants used throughout the application:
- Application titles and labels
- Model hyperparameters and default values
- Data paths and class names

#### `src/data/`
Handles data loading and preprocessing:
- `load_data()`: Loads and label-encodes the mushroom dataset
- `split_data()`: Splits data into training and testing sets

#### `src/models/`
Contains the `ModelTrainer` class for model training and evaluation:
- `train_svm()`: Trains Support Vector Machine
- `train_logistic_regression()`: Trains Logistic Regression
- `train_random_forest()`: Trains Random Forest
- Returns performance metrics (accuracy, precision, recall)

#### `src/utils/`
Utility functions for visualization:
- `plot_metrics()`: Plots confusion matrix, ROC curve, and precision-recall curve
- `display_metrics()`: Displays model performance metrics

#### `src/components/`
UI components for the Streamlit interface:
- `sidebar.py`: Renders sidebar with classifier selection and hyperparameters
- `main_content.py`: Renders main content area with classification results

#### `app.py`
Main entry point that orchestrates all components and runs the application.

##  Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/pratyushsrivastava500/Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not.git
   cd Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib
   ```

##  Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Using the application**:
   
   **Step 1**: Choose a classifier from the sidebar
   - Support Vector Machine (SVM)
   - Logistic Regression
   - Random Forest
   
   **Step 2**: Tune the hyperparameters
   - Adjust sliders and input fields based on the selected model
   
   **Step 3**: Select performance metrics to visualize
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve
   
   **Step 4**: Click "Classify" button
   - View model accuracy, precision, and recall
   - Analyze the selected performance metrics
   
   **Optional**: Check "Show raw data" to explore the dataset

##  Extending the Application

Thanks to the modular architecture, you can easily extend the application:

### Adding a New Classifier

1. **Add configuration** in `src/config/config.py`:
   ```python
   CLASSIFIER_NEW = "New Classifier Name"
   AVAILABLE_CLASSIFIERS.append(CLASSIFIER_NEW)
   ```

2. **Add training method** in `src/models/model_trainer.py`:
   ```python
   def train_new_classifier(self, param1, param2):
       self.model = NewClassifier(param1=param1, param2=param2)
       self.model.fit(self.x_train, self.y_train)
       return self._evaluate_model()
   ```

3. **Add UI controls** in `src/components/sidebar.py`:
   ```python
   def _get_new_classifier_params():
       st.sidebar.subheader("Model Hyperparameters")
       param1 = st.sidebar.slider("Parameter 1", ...)
       return {'param1': param1, 'param2': param2}
   ```

4. **Update main content** in `src/components/main_content.py`:
   ```python
   elif classifier == config.CLASSIFIER_NEW:
       st.subheader("New Classifier Results")
       results = trainer.train_new_classifier(...)
   ```

### Adding a New Metric

1. **Add to config** in `src/config/config.py`:
   ```python
   AVAILABLE_METRICS.append('New Metric')
   ```

2. **Add plotting function** in `src/utils/visualization.py`:
   ```python
   if 'New Metric' in metrics_list:
       st.subheader("New Metric")
       # Your plotting code
       st.pyplot()
   ```

### Modifying Data Processing

Simply update the functions in `src/data/data_loader.py` without affecting other modules.

### Updating Configuration

All constants and settings are centralized in `src/config/config.py` for easy modification.

##  Model Performance Metrics

### Accuracy
Measures the overall correctness of the model:
```
Accuracy = (True Positives + True Negatives) / Total Samples
```

### Precision
Indicates how many predicted poisonous mushrooms are actually poisonous:
```
Precision = True Positives / (True Positives + False Positives)
```

### Recall (Sensitivity)
Shows how many actual poisonous mushrooms were correctly identified:
```
Recall = True Positives / (True Positives + False Negatives)
```

### Confusion Matrix
Visual representation of prediction results showing:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

### ROC Curve
Illustrates the diagnostic ability of the classifier at various threshold settings.

### Precision-Recall Curve
Shows the tradeoff between precision and recall for different threshold values.

##  Technologies Used

- **Python**: Core programming language (3.7+)
- **Streamlit**: Web application framework for ML/data science
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library
  - SVM (Support Vector Machine)
  - Logistic Regression
  - Random Forest Classifier
  - Label Encoder
  - Train-Test Split
  - Performance Metrics
- **Matplotlib**: Data visualization for plots and charts

### Code Architecture
- **Modular Design**: Organized into separate modules for configuration, data handling, models, utilities, and UI components
- **Object-Oriented**: Uses classes for model training and evaluation
- **Functional Programming**: Separate functions for data loading, preprocessing, and visualization
- **Caching**: Uses Streamlit's `@st.cache_data` decorator for performance optimization

##  Screenshots

*Add screenshots of your application here showing:*
- Main interface with classifier selection
- Hyperparameter tuning panel
- Model results with metrics
- Confusion matrix visualization
- ROC and Precision-Recall curves

##  Future Enhancements

- [ ] Add more classification algorithms (XGBoost, Neural Networks, Gradient Boosting)
- [ ] Implement feature importance visualization
- [ ] Add cross-validation for more robust evaluation
- [ ] Include confidence intervals for predictions
- [ ] Deploy on cloud platform (Streamlit Cloud, Heroku, AWS)
- [ ] Add model comparison feature (side-by-side comparison)
- [ ] Implement data preprocessing options (normalization, standardization)
- [ ] Create downloadable model reports (PDF/HTML)
- [ ] Add A/B testing between models
- [ ] Include mushroom image classification using CNN
- [ ] Add educational tooltips for ML concepts
- [ ] Implement model saving/loading functionality
- [ ] Add unit tests for all modules
- [ ] Create API endpoints for model predictions
- [ ] Implement logging and error handling
- [ ] Add data validation and quality checks

##  Important Note

**This application is for educational and demonstration purposes only.** 

 **DO NOT** use this application to determine if real mushrooms are safe to eat. Consuming wild mushrooms can be extremely dangerous and potentially fatal. Always consult with mycology experts and use proper field guides when foraging for mushrooms.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is open source and available under the MIT License.

##  Author

**Pratyush Srivastava**
- GitHub: [@pratyushsrivastava500](https://github.com/pratyushsrivastava500)

##  Acknowledgments

- **UCI Machine Learning Repository** for providing the mushroom dataset
- **Streamlit** community for excellent documentation and support
- **Scikit-learn** for comprehensive machine learning tools
- Mushroom data contributors and researchers

##  References

- [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

 **If you found this project helpful, please give it a star!**

 **Learning Opportunity**: This project is perfect for understanding binary classification, model comparison, and interactive ML applications.

---

**Disclaimer**: This is a learning project. Model predictions are based on historical data and should not be used for actual mushroom identification in the wild.
