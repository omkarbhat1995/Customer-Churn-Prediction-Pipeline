# modeling.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

def train_and_evaluate(X_train, y_train, X_test, y_test):
    print("Starting Hyperparameter Tuning...")
    
    # ---------------------------------------------------------
    # 1. Tune Logistic Regression
    # ---------------------------------------------------------
    y_train_flat = np.ravel(y_train)
    print("\nTuning Logistic Regression...")
    log_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],            # Inverse of regularization strength
        'l1_ratio': [0, 1],                      # 1 acts as 'l1' penalty, 0 acts as 'l2' penalty
        'solver': ['saga']                       # 'saga' is required to support l1_ratio
    }
    
    # cv=5 means 5-fold cross-validation
    log_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), # Added max_iter=1000 for saga convergence
                            log_param_grid, 
                            cv=5, 
                            scoring='recall',    
                            n_jobs=-1)           
    log_grid.fit(X_train, y_train_flat)
    
    best_log_model = log_grid.best_estimator_
    y_pred_log = best_log_model.predict(X_test)
    
    print(f"Best Logistic Regression Params: {log_grid.best_params_}")
    print("--- Logistic Regression (Tuned) ---")
    print(classification_report(y_test, y_pred_log))
    
    # ---------------------------------------------------------
    # 2. Tune Decision Tree
    # ---------------------------------------------------------
    print("\nTuning Decision Tree...")
    dt_param_grid = {
        'max_depth': [5, 10, 15, 20, None],       # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],          # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],            # Minimum samples required to be at a leaf node
        'criterion': ['gini', 'entropy']          # Function to measure the quality of a split
    }
    
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                           dt_param_grid, 
                           cv=5, 
                           scoring='recall',      # Prioritizing Recall for Churn
                           n_jobs=-1)
    dt_grid.fit(X_train, y_train_flat)
    
    best_dt_model = dt_grid.best_estimator_
    y_pred_dt = best_dt_model.predict(X_test)
    
    print(f"Best Decision Tree Params: {dt_grid.best_params_}")
    print("--- Decision Tree (Tuned) ---")
    print(classification_report(y_test, y_pred_dt))
    
    # ---------------------------------------------------------#
    # 3. Save the Best Overall Model
    # ---------------------------------------------------------#
    # Dynamically select the model with the highest test accuracy to save
    log_acc = accuracy_score(y_test, y_pred_log)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    joblib.dump(best_dt_model, 'models//dt_model.pkl')
    joblib.dump(best_log_model, 'models//log_model.pkl')

    if dt_acc >= log_acc:
        print(f"\nSaving Tuned Decision Tree as best model (Test Accuracy: {dt_acc:.4f})")
        joblib.dump(best_dt_model, 'models//best_model.pkl')
    else:
        print(f"\nSaving Tuned Logistic Regression as best model (Test Accuracy: {log_acc:.4f})")
        joblib.dump(best_log_model, 'models//best_model.pkl')

if __name__ == "__main__":
    # Quick test execution block
    print("This module is intended to be imported, but can be tested with dummy data.")
    import pandas as pd
    from transformation import transform_data
    
    X_train = pd.read_csv('data\\transformed\\X_train_balanced.csv')
    X_test = pd.read_csv('data\\transformed\\X_test_scaled.csv')
    y_train = pd.read_csv('data\\transformed\\y_train_balanced.csv')
    y_test = pd.read_csv('data\\transformed\\y_test.csv')

    train_and_evaluate(X_train, y_train, X_test, y_test)
