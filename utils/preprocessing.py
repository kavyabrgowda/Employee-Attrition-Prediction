import joblib
import pandas as pd


def preprocess_input(data_dict):

    encoders = joblib.load("models/label_encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    df = pd.DataFrame([data_dict])

    # Encode categorical columns
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # Ensure correct column order
    df = df[feature_columns]

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled


# ✅ BULK PREPROCESSING (Matching same logic)
def preprocess_bulk(df):

    encoders = joblib.load("models/label_encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    # Encode categorical columns
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # Ensure same column order
    df = df[feature_columns]

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled