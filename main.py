# main.py
from config import RAW_DATA_PATH
from cleaning import load_and_clean_data
from transformation import transform_data
from modeling import train_and_evaluate
from visualization import generate_evaluation_report_and_plots
import os

def run_pipeline():
    print("Starting ML Pipeline...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Please place {RAW_DATA_PATH} in the root directory.")
        return
    for folder in ['data/clean', 'data/transformed', 'models', 'visualizations']:
        os.makedirs(folder, exist_ok=True)
    # 1. Cleaning
    df_clean = load_and_clean_data(RAW_DATA_PATH)
    
    # 2. Transformation
    transformed_data = transform_data(df_clean, 'churn')
    if transformed_data is None:
        print("Error: Data transformation failed.")
        return
    X_train, X_test, y_train, y_test = transformed_data
    
    # 3. Modeling
    train_and_evaluate(X_train, y_train, X_test, y_test)

    # 4. Evaluation and Visualization (Optional)
    generate_evaluation_report_and_plots(X_test, y_test, df_clean)

if __name__ == "__main__":
    run_pipeline()
