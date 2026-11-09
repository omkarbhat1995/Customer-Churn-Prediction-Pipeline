# Customer Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)

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
  
## **Results and Insights:**
   Based on our model evaluation, we discovered key insights regarding model selection and customer behavior:

   ### Model Performance Comparison
   *   **Logistic Regression:** Achieved an overall accuracy of 59.73% and an AUC of 0.667. While its accuracy is lower, it successfully identified 64.60% of all churning customers (Recall). 
   *   **Decision Tree:** Achieved a higher overall accuracy of 80.77%, but a much lower AUC of 0.519. It heavily favored the majority class and only identified 15.95% of churning customers.

   ### Key Drivers of Churn
   Our feature importance analysis revealed that customer service interactions are the strongest predictors of churn. The top three contributing factors are:
   1.  **Number of Service Calls:** The leading indicator of customer attrition.
   2.  **Number of Complaints:** The second highest contributing feature.
   3.  **Contract Type:** The third most critical factor influencing retention.

   ### Business Recommendations
   For a proactive churn prevention campaign, we recommend deploying the **Logistic Regression** model. In churn prediction, a False Negative (failing to identify a churning customer) is significantly more costly than a False Positive (offering a retention discount to a happy customer). The Logistic Regression model's superior recall ensures we capture and intervene with the maximum number of at-risk customers. Retention strategies should heavily target customers experiencing high volumes of service calls and complaints.

  ### Insights from Plots:
  * **Churn Distribution:**
      This plot highlights a severe class imbalance in the original dataset, with 90% of customers remaining and only 10% churning, necessitating the use of SMOTE during our transformation phase to prevent the models from becoming biased toward the majority class.
    
    <img width="567" height="397" alt="plot_churn_distribution" src="https://github.com/user-attachments/assets/9bcd6e81-99b6-41b9-a413-07e4bcc7d501" />

  * **Confusion Matrices:**
    This comparison of predicted versus actual classifications highlights that Logistic Regression correctly identifies far more actual churners (higher recall) than the Decision Tree, which heavily over-predicts the "No Churn" majority class.
    
    <img width="1184" height="484" alt="plot_confusion_matrices" src="https://github.com/user-attachments/assets/b4fc7a0f-559f-49bd-80d2-d042c0ac815f" />

  * **Feature Importance:**
      This chart ranks the variables driving the Decision Tree's predictions, clearly showing that the number of customer service calls and complaints are the strongest leading indicators of churn.

   <img width="1010" height="551" alt="plot_feature_importance" src="https://github.com/user-attachments/assets/3f7a463d-dda0-4fe2-874d-2e8b384803fa" />

  * **ROC Curve:**
      This plot illustrates the trade-off between true positive and false positive rates, demonstrating that Logistic Regression (AUC = 0.667) has a better overall ability to distinguish between churn and non-churn customers compared to the Decision Tree (AUC = 0.519).
    
    <img width="694" height="551" alt="plot_roc_curve" src="https://github.com/user-attachments/assets/ea0c5f41-0a9f-4c6a-ab3f-4151b456dd1c" />


## 📈 Evaluation Metrics Focus

In the context of customer churn, **Recall** is heavily prioritized during model tuning. Failing to identify a churning customer (False Negative) is generally more costly than incorrectly predicting a loyal customer might churn (False Positive).
