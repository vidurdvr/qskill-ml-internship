import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_housing_data() -> pd.DataFrame:
    """
    Fetch the California Housing dataset from scikit-learn.
    Returns a pandas DataFrame with feature columns and 'MedHouseVal' as target.
    
    Target unit: Hundreds of thousands of dollars ($100,000s).
    """
    california = fetch_california_housing(as_frame=True)
    df = california.frame.copy()
    return df

if __name__ == "__main__":
    df = load_housing_data()
    print("California Housing dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nFeature info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
