# cleaning.py
import pandas as pd

def load_and_clean_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    df_clean = df.copy()
        
    # 1. Drop irrelevant identifier columns (looks like 'customer_' is an ID)
    cols_to_drop = [col for col in df_clean.columns if 'customer_' in col.lower()]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    # 2. Handle Date Columns (e.g., 'signup_dat')
    if 'signup_dat' in df_clean.columns:
        # Option A: Drop it if you are using 'tenure' instead
        df_clean = df_clean.drop(columns=['signup_dat'])
        # Option B: Convert to datetime if you want to extract features
        # df_clean['signup_dat'] = pd.to_datetime(df_clean['signup_dat'])
    
    # 3. Handle Missing Values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Fill missing categoricals with the mode (most frequent value)
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            # Fill missing numericals with the median to avoid outlier skew
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
    # 4. Standardize text data (optional but recommended)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].str.lower().str.strip()

    df_clean.to_csv('data/clean/customer_churn_cleaned.csv', index=False)
    return df_clean

if __name__ == "__main__":
    # For testing purposes
    from config import RAW_DATA_PATH, CLEANED_DATA_PATH
    df_cleaned = load_and_clean_data(RAW_DATA_PATH)
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")
