import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def preprocess_input(data_dict):

    df = pd.DataFrame([data_dict])

    # remove ID
    df = df.drop(columns=["Employee_ID"], errors="ignore")

    return df