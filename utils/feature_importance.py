import joblib
import pandas as pd

def get_top_features():

    model = joblib.load("models/model.pkl")

    features = [
        "Age", "MaritalStatus", "DistanceFromHome",
        "Department", "JobRole", "BusinessTravel",
        "JobSatisfaction", "YearsAtCompany",
        "YearsSinceLastPromotion", "MonthlyIncome"
    ]

    importance = model.coef_[0]

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return df.head(3)