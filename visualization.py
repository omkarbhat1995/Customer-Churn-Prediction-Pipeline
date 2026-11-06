# visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import joblib
import os

def generate_evaluation_report_and_plots(X_test, y_test, df_clean):
    print("\n--- Generating Evaluation Report & Visualizations ---")
    
    # Load Models
    log_model = joblib.load('log_model.pkl')
    dt_model = joblib.load('dt_model.pkl')
    
    models = {
        'Logistic Regression': log_model,
        'Decision Tree': dt_model
    }
    
    # 1. Calculate Metrics & Print Report
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC':<10}")
    print("-" * 85)
    
    roc_data = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {roc_auc:<10.4f}")

    # Set up visualization grid
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # Plot 1: Churn Distribution
    # ==========================================
    plt.figure(figsize=(6, 4))
    sns.countplot(x='churn', data=df_clean, hue='churn', palette='Set2', legend=False)
    plt.title('Churn Distribution (Original Cleaned Data)')
    plt.xlabel('Churn (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig('visualizations\\plot_churn_distribution.png', bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Plot 2: ROC Curve (Combined)
    # ==========================================
    plt.figure(figsize=(8, 6))
    for name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('visualizations\\plot_roc_curve.png', bbox_inches='tight')
    plt.close()

    # ==========================================
    # Plot 3: Confusion Matrices
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_yticklabels(['No Churn', 'Churn'])
        
    plt.tight_layout()
    plt.savefig('visualizations\\plot_confusion_matrices.png', bbox_inches='tight')
    plt.close()

    # ==========================================
    # Plot 4: Feature Importance (Decision Tree)
    # ==========================================
    feature_names = dt_model.feature_names_in_
    importance = dt_model.feature_importances_
    
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    # Added hue and legend=False here as well to prevent future Seaborn warnings on barplot
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', palette='viridis', legend=False)
    plt.title('Top 10 Feature Importances (Decision Tree)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.savefig('visualizations\\plot_feature_importance.png', bbox_inches='tight')
    plt.close()

    print("\nVisualizations saved as PNG files in the current directory.")

if __name__ == "__main__":
    # Quick test execution block
    print("This module is intended to be imported, but can be tested with dummy data.")
    import pandas as pd
    df_clean = pd.read_csv('data/clean/customer_churn_cleaned.csv')
    X_test = pd.read_csv('data/transformed/X_test_scaled.csv')
    y_test = pd.read_csv('data/transformed/y_test.csv').squeeze()  # Ensure y_test is a Series
    generate_evaluation_report_and_plots(X_test, y_test, df_clean)
