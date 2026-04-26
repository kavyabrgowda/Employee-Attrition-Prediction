import joblib
import pandas as pd

def get_top_features():

    model = joblib.load("models/model.pkl")

    features = joblib.load("models/feature_columns.pkl")

    importance = model.coef_[0]

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return df.head(3)