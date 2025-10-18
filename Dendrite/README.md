# ML Pipeline - JSON-Driven Machine Learning

A production-ready machine learning pipeline that executes complete ML workflows from JSON configuration files. Built with scikit-learn.

## Features

- JSON-driven configuration
- Automatic preprocessing for numerical and categorical features
- Feature engineering capabilities
- Multiple feature reduction methods
- Support for training multiple models simultaneously
- Comprehensive metrics reporting
- Integrated hyperparameter optimization

## Installation

```
# Clone repository
cd ml-pipeline

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```
python [main.py](http://main.py) <config.json> <data.csv>
```

Example:

```
python [main.py](http://main.py) data/algoparams_from_ui.json data/iris_modified.csv
```

## Supported Models

### Regression

- Linear Regression
- Ridge & Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor

### Classification

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier
- K-Nearest Neighbors Classifier
- Gaussian Naive Bayes

## Configuration

The pipeline uses JSON configuration files with the following structure:

```json
{
  "design_state_data": {
    "target": {
      "target": "column_name",
      "prediction_type": "Regression"
    },
    "feature_handling": { ... },
    "feature_generation": { ... },
    "feature_reduction": {
      "feature_reduction_method": "Tree-based",
      "num_of_features_to_keep": "4"
    },
    "algorithms": {
      "RandomForestRegressor": {
        "is_selected": true,
        "min_trees": 10,
        "max_trees": 20
      }
    }
  }
}
```

## Requirements

- Python >= 3.7
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
