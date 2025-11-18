# 🍄 Mushroom Classification Web App

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> A machine learning web application that classifies mushrooms as edible or poisonous using SVM, Logistic Regression, and Random Forest algorithms. Built with clean modular architecture and interactive visualizations.

## 📋 Overview

The Mushroom Classification Web App enables users to:

- **Classify Mushrooms** as edible or poisonous with high accuracy
- **Choose ML Algorithms** from SVM, Logistic Regression, or Random Forest
- **Tune Hyperparameters** in real-time for optimal performance
- **Visualize Metrics** with confusion matrices, ROC curves, and precision-recall curves
- **Explore Dataset** with 8,124 samples across 23 features

## ✨ Features

### 🎯 ML-Powered Classification
- Binary classification with 3 powerful algorithms
- Real-time predictions with sub-second response
- Supports 23 mushroom characteristics
- Accuracy scores up to 100%

### 🏗️ Clean Architecture
- Modular design with separation of concerns
- Type hints and comprehensive docstrings
- Centralized configuration management
- Production-ready error handling

### 💻 User Experience
- Clean, intuitive Streamlit interface
- Interactive hyperparameter tuning
- Multiple performance visualizations
- Data exploration capabilities

### 📊 Advanced Visualizations
- Confusion Matrix for classification analysis
- ROC Curve for model comparison
- Precision-Recall Curve for threshold tuning
- Real-time metric updates

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pratyushsrivastava500/Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not.git
   cd Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:8501`

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

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
│  • Classifier selection             │
│  • Hyperparameter tuning            │
│  • Display predictions & metrics    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│       Component Layer               │
│  • sidebar.py (UI controls)         │
│  • main_content.py (display)        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Model Layer                  │
│  • ModelTrainer class               │
│  • train_svm()                      │
│  • train_logistic_regression()      │
│  • train_random_forest()            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Utility & Visualization Layer    │
│  • plot_metrics()                   │
│  • display_metrics()                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Data Processing Layer            │
│  • load_data()                      │
│  • split_data()                     │
│  • Label encoding                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      Configuration Layer            │
│  • Paths & parameters               │
│  • Model hyperparameters            │
└─────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.0+ |
| **ML Models** | Scikit-learn (SVM, LR, RF) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Data Encoding** | LabelEncoder |

## 📁 Project Structure

```
Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore patterns
├── README.md                     # Project documentation
│
├── data/                         # Data directory
│   └── mushrooms.csv            # Mushroom classification dataset
│
└── src/                          # Source code directory
    ├── __init__.py              # Package initialization
    │
    ├── config/                   # Configuration module
    │   ├── __init__.py
    │   └── config.py            # Application settings and constants
    │
    ├── data/                     # Data handling module
    │   ├── __init__.py
    │   └── data_loader.py       # Data loading and preprocessing
    │
    ├── models/                   # Model training module
    │   ├── __init__.py
    │   └── model_trainer.py     # Model training and evaluation
    │
    ├── utils/                    # Utility functions
    │   ├── __init__.py
    │   └── visualization.py     # Plotting and visualization
    │
    └── components/               # UI components
        ├── __init__.py
        ├── sidebar.py           # Sidebar component
        └── main_content.py      # Main content component
```

## 📊 Dataset Information

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom)

**Statistics:**

| Attribute | Details |
|-----------|---------|
| **Records** | 8,124 mushroom samples |
| **Features** | 23 categorical attributes |
| **Target Variable** | Edible (e) or Poisonous (p) |
| **Species** | 23 species from Agaricus and Lepiota families |

**Key Features:**

| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| `type` | Target variable | Binary | Edible (0), Poisonous (1) |
| `cap_shape` | Shape of mushroom cap | Categorical | Bell, Conical, Convex |
| `cap_surface` | Surface texture | Categorical | Fibrous, Grooves, Scaly |
| `cap_color` | Color of cap | Categorical | Brown, Gray, Red, White |
| `odor` | Mushroom smell | Categorical | Almond, Anise, Foul, None |
| `gill_size` | Size of gills | Categorical | Broad, Narrow |
| `stalk_shape` | Shape of stalk | Categorical | Enlarging, Tapering |
| `habitat` | Growing environment | Categorical | Grasses, Leaves, Woods |

**Preprocessing Steps:**
- Label encoding applied to all categorical features
- Train-test split (70-30 ratio)
- No missing values or duplicates
- All features normalized through encoding

## 📖 Usage Guide

### Making Classifications

1. **Select Classifier:**
   - Choose from SVM, Logistic Regression, or Random Forest
   
2. **Tune Hyperparameters:**
   - Adjust regularization parameters (C)
   - Select kernel type (for SVM)
   - Set number of estimators and max depth (for Random Forest)
   - Configure max iterations (for Logistic Regression)

3. **Choose Metrics:**
   - Select visualizations (Confusion Matrix, ROC Curve, Precision-Recall Curve)

4. **Click "Classify":**
   - View accuracy, precision, and recall scores
   - Analyze selected performance visualizations

5. **Explore Data:**
   - Check "Show raw data" to view the dataset

### Example Usage

**Support Vector Machine (SVM):**
```
Classifier: SVM
C: 1.0
Kernel: RBF
Gamma: scale
Result: 100% Accuracy
```

**Random Forest:**
```
Classifier: Random Forest
Estimators: 100
Max Depth: 10
Bootstrap: True
Result: 100% Accuracy
```

## 🤖 Model Performance

**Algorithms:** SVM, Logistic Regression, Random Forest

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| **SVM (RBF)** | 100% | 1.00 | 1.00 |
| **Logistic Regression** | ~95% | 0.95 | 0.95 |
| **Random Forest** | 100% | 1.00 | 1.00 |

**Top Predictive Features:**
1. Odor (most significant)
2. Spore Print Color
3. Gill Size
4. Gill Color
5. Ring Type

## 🔮 Future Enhancements

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

## 🔮 Future Enhancements

- [ ] Add more ML models (XGBoost, Neural Networks, Gradient Boosting)
- [ ] Implement hyperparameter optimization (Grid Search, Random Search)
- [ ] Add cross-validation for robust evaluation
- [ ] Feature importance visualization
- [ ] Deploy to cloud (Streamlit Cloud/Heroku/AWS)
- [ ] Add model comparison dashboard
- [ ] Implement model saving/loading functionality
- [ ] Create REST API endpoints
- [ ] Add unit tests for all modules
- [ ] Mushroom image classification using CNN
- [ ] Real-time prediction API
- [ ] Mobile app version

## 🔧 Troubleshooting

**Issue: Streamlit not found**
```bash
pip install streamlit
```

**Issue: Module import errors**
```bash
pip install -r requirements.txt
```

**Issue: sklearn metrics errors**
```bash
pip install --upgrade scikit-learn
```

**Issue: Visualization not showing**
```bash
pip install matplotlib
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the mushroom dataset
- **Streamlit** community for excellent documentation and support
- **Scikit-learn** for comprehensive machine learning tools
- Mushroom data contributors and researchers

## 📧 Contact

For questions or support, please open an issue on GitHub.

⚠️ **Disclaimer:** This application is for educational and demonstration purposes only. **DO NOT** use this application to determine if real mushrooms are safe to eat. Consuming wild mushrooms can be extremely dangerous and potentially fatal. Always consult with mycology experts and use proper field guides when foraging for mushrooms.

---

<div align="center">

**Made with ❤️ and Python | © 2025 Pratyush Srivastava**

**[GitHub](https://github.com/pratyushsrivastava500) | [Repository](https://github.com/pratyushsrivastava500/Web-App-using-Streamlit-that-Predict-Weather-the-Mushoom-are-posionsous-or-not)**

</div>
