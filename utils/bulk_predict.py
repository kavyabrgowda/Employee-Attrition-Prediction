import joblib
import pandas as pd
from utils.preprocessing import preprocess_input

def bulk_predict(df):

    model = joblib.load("models/model.pkl")

    predictions = []
    probabilities = []

    for _, row in df.iterrows():
        processed = preprocess_input(row.to_dict())
        prob = model.predict_proba(processed)[0][1]

        probabilities.append(round(prob * 100, 2))

        if prob > 0.6:
            predictions.append("High Risk")
        elif prob > 0.4:
            predictions.append("Medium Risk")
        else:
            predictions.append("Low Risk")

    df["Attrition Probability (%)"] = probabilities
    df["Risk"] = predictions

    output_path = "data/bulk_predictions.csv"
    df.to_csv(output_path, index=False)

    return output_path