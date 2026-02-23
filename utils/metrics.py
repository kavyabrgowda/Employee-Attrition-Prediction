import pandas as pd
import joblib
from utils.preprocessing import preprocess_input

def dashboard_metrics(csv_path):

    df = pd.read_csv(csv_path)
    model = joblib.load("models/model.pkl")

    probabilities = []
    risks = []

    for _, row in df.iterrows():
        processed = preprocess_input(row.to_dict())
        prob = model.predict_proba(processed)[0][1]
        probabilities.append(prob)

        if prob > 0.6:
            risks.append("High Risk")
        elif prob > 0.4:
            risks.append("Medium Risk")
        else:
            risks.append("Low Risk")

    df["Probability"] = probabilities
    df["Risk"] = risks

    total = len(df)
    high = (df["Risk"] == "High Risk").sum()
    medium = (df["Risk"] == "Medium Risk").sum()
    low = (df["Risk"] == "Low Risk").sum()

    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "risk_percent": round((high / total) * 100, 2),
        "avg_income": round(df["MonthlyIncome"].mean(), 2),
        "avg_satisfaction": round(df["JobSatisfaction"].mean(), 2),
        "avg_tenure": round(df["YearsAtCompany"].mean(), 2)
    }