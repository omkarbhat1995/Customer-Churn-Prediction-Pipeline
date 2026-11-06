# Customer Churn Prediction Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict customer churn. It takes raw customer data, cleans it, balances the target classes using SMOTE, tunes multiple machine learning models (Logistic Regression and Decision Tree) using `GridSearchCV`, and generates a comprehensive evaluation report with visualizations.

## 📊 Data Source
The dataset used in this project is the **[Customer Churn Prediction Dataset (1M)](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m?resource=download)** available on Kaggle. 

To run this pipeline locally, you must download the dataset from Kaggle and place it in the `data/raw/` directory.

## 🚀 Features

* **Modular Architecture:** Clean, maintainable code split across distinct stages (Cleaning, Transformation, Modeling, Visualization).

* **Automated Preprocessing:** Handles missing values (median/mode imputation) and automatically label-encodes categorical features.

* **Imbalance Handling:** Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to ensure the models effectively learn from minority churn cases.

* **Hyperparameter Tuning:** Implements `GridSearchCV` to automatically find the optimal parameters for both Logistic Regression (using the `saga` solver) and Decision Tree classifiers.

* **Rich Visualizations:** Automatically generates and saves plots for Churn Distribution, Feature Importance, Confusion Matrices, and ROC/AUC curves.

## 📂 Project Structure

```
├── data/
│   ├── raw/                 # Place your Kaggle dataset here (e.g., customer_churn_1M.csv)
│   ├── clean/               # Cleaned data is saved here (Ignored by Git)
│   └── transformed/         # Scaled, encoded, and SMOTE-balanced data (Ignored by Git)
├── models/                  # Saved models (.pkl) and preprocessing artifacts (Ignored by Git)
├── visualizations/          # Generated evaluation plots (.png)
├── config.py                # Global configuration and file paths
├── cleaning.py              # Data cleaning and imputation logic
├── transformation.py        # Encoding, scaling, and SMOTE implementation
├── modeling.py              # Model training, GridSearch tuning, and saving
├── visualization.py         # Model evaluation metrics and matplotlib/seaborn plots
├── main.py                  # Main orchestration script to run the full pipeline
└── requirements.txt         # Python dependencies

```

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/customer-churn-pipeline.git
   cd customer-churn-pipeline
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. **Add your data:**
   Download the dataset from the Kaggle link above, place it into the `data/raw/` directory, and ensure it is named correctly according to `config.py` (default: `customer_churn_1M.csv`).

2. **Run the pipeline:**
   Execute the main script from the root directory:

   ```bash
   python main.py
   ```

3. **Review the Results:**

   * **Terminal Output:** View the cross-validated classification reports, best hyperparameters found, and the overall accuracy/precision/recall/F1/AUC scores.
   * **Visualizations:** Check the `visualizations/` folder for the generated `.png` graphs (ROC Curve, Confusion Matrices, Feature Importances, etc.).
   * **Models:** The best performing models and the `preprocessor.pkl` (containing your fitted scalers and encoders) will be saved in the `models/` folder, ready for deployment or inference on new data.

## 📈 Evaluation Metrics Focus

In the context of customer churn, **Recall** is heavily prioritized during model tuning. Failing to identify a churning customer (False Negative) is generally more costly than incorrectly predicting a loyal customer might churn (False Positive).