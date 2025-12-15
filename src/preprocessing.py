import pandas as pd

def clean_data(df):
    df = df.copy()

    # Remove unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Standardize text columns
    df["fuel_type"] = df["fuel_type"].str.strip().str.title()
    df["transmission_type"] = df["transmission_type"].str.strip().str.title()
    df["seller_type"] = df["seller_type"].str.strip().str.title()

    # Convert all categorical to string type
    cat_cols = ["brand", "model", "fuel_type", "transmission_type", "seller_type"]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/cardekho_dataset.csv")
    df = clean_data(df)
    print(df.head())
    print(df.info())
