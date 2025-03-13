# Crime Prediction and Fairness Analysis Project

## Project Description

This project aims to build a machine learning model to predict crime scores in communities based on socio-economic and demographic data, while analyzing the **fairness** of the model with respect to sensitive attributes such as race and ethnicity. The model predicts a crime score, and the **fairness** is assessed using the **AIF360** framework, which evaluates the model's fairness by considering the differences in treatment between sensitive groups (e.g., racial groups).

## Objectives

1. **Build a prediction model**: Develop a predictive model to estimate a crime score based on socio-economic data.
2. **Fairness analysis**: Use the **AIF360** package to evaluate the fairness of the model with respect to protected groups (race and ethnicity).
3. **Optimization**: Experiment with different machine learning techniques to improve the model's performance while minimizing bias related to fairness.

---

## Methodology

### 1. Data Preprocessing

- **Sensitive Attributes**: The sensitive attributes, such as race and ethnicity, are extracted as separate variables.
- **Dimensionality Reduction**: To handle the complexity of the data, **Principal Component Analysis (PCA)** is performed, reducing the dataset's dimensionality while retaining the relevant information.
- **Separation of Sensitive Attributes**: Sensitive attributes are separated for a precise fairness evaluation of the model.

### 2. Model Training

We experimented with several models to predict the crime score:

- **Random Forest**: A baseline regression model.
- **XGBoost**: A more advanced model using boosted decision trees for better performance.
- **Grid Search Optimization**: A search for the best hyperparameters to optimize the XGBoost model.

### 3. Fairness Analysis

Once the model is trained, a **fairness analysis** is performed using **AIF360**. The following metrics are calculated to assess bias between protected and unprotected groups:
- **Disparate Impact**: Measures the disparity in treatment between groups.
- **Mean Difference**: The average difference in predictions for protected and unprotected groups.
- **Statistical Parity Difference**: The difference in the probability of prediction between groups.
- **Equal Opportunity Difference** and **Equalized Odds Difference**: Evaluates inequalities in model performance between groups.

### 4. Optimization and Testing

The goal is to find a balance between model performance (measured by **RMSE** and **R²**) and fairness of predictions by tuning hyperparameters and using techniques to minimize bias in results.

---

## Project Structure

Here’s the directory and file structure of the project:

```
/Communities_and_Crime
│
├── data/                          # Contains preprocessed datasets
│   ├── X_train_pca.npy             # Training data (reduced via PCA)
│   ├── X_test_pca.npy              # Test data (reduced via PCA)
│   ├── y_train.npy                 # Training labels
│   ├── y_test.npy                  # Test labels
│   ├── X_sensitive_train.npy       # Sensitive attributes for training data
│   └── X_sensitive_test.npy        # Sensitive attributes for test data
│
├── script/                         # Contains scripts for preprocessing, training, etc.
│   ├── preprocess.py               # Data preprocessing: PCA and sensitive attribute separation
│   ├── model.py                    # Model training (Random Forest, XGBoost)
│   ├── fairness.py                 # Fairness analysis using AIF360
│   └── evaluate.py                 # Model performance and fairness evaluation
│
├── models/                         # Contains trained models
│   └── best_xgboost_model.pkl      # Optimized XGBoost model
│
├── saves/
│   └── *.png                       # Diagram for visualization of fairness
│
│
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## Usage Instructions

### Prerequisites

- **Python 3.8+**
- **Required Packages**: Use `pip` to install dependencies listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `numpy`, `pandas`, `scikit-learn`, `xgboost` for data processing and model training.
- `aif360` for fairness analysis of the model.
- `joblib` for saving and loading models.

### Running the Code

1. **Data Preprocessing**: Run the `preprocess.py` script to perform PCA, separate sensitive attributes, and save the processed data.

```bash
python script/preprocess.py
```

2. **Model Training**: Run the `model.py` script to train the regression models (`RandomForest` and `XGBoost`) and perform a grid search for the best hyperparameters for `XGBoost`.

```bash
python script/model.py
```

3. **Model Performance and Fairness Evaluation**: Run the `fairness.py` script to evaluate the model's performance (RMSE, R²) and analyze its fairness (Disparate Impact, etc.).

```bash
python script/fairness.py
```

---