# transformation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

def transform_data(df, target_col='churn'):
    print("Transforming data...")
    df_transformed = df.copy()

    if target_col not in df_transformed.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")


    label_encoders = {}
    categorical_cols = df_transformed.select_dtypes(include=['object']).columns
    if target_col in categorical_cols:
        categorical_cols = categorical_cols.drop(target_col)
        df_transformed[target_col] = df_transformed[target_col].map(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)    
        
    for col in categorical_cols:
        le = LabelEncoder()
        df_transformed[col] = le.fit_transform(df_transformed[col])
        label_encoders[col] = le

    X = df_transformed.drop(target_col, axis=1)
    y = df_transformed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train.to_csv('data/clean/X_train.csv', index=False)
    X_test.to_csv('data/clean/X_test.csv', index=False)
    y_train.to_csv('data/clean/y_train.csv', index=False)
    y_test.to_csv('data/clean/y_test.csv', index=False)

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("Applying SMOTE to balance the training classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    joblib.dump({
        'scaler': scaler, 
        'label_encoders': label_encoders,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }, 'models//preprocessor.pkl')
    
    print(f"Original training shape: {X_train_scaled.shape}, Balanced training shape: {X_train_bal.shape}")
    X_train_bal.to_csv('data/transformed/X_train_balanced.csv', index=False)
    X_test_scaled.to_csv('data/transformed/X_test_scaled.csv', index=False)
    y_train_bal.to_csv('data/transformed/y_train_balanced.csv', index=False)
    y_test.to_csv('data/transformed/y_test.csv', index=False)
    return X_train_bal, X_test_scaled, y_train_bal, y_test

if __name__ == "__main__":
    # Quick test execution block
    print("This module is intended to be imported, but can be tested with a dummy DataFrame.")
    df = pd.read_csv('data\\clean\\customer_churn_cleaned.csv')
    transform_data(df)