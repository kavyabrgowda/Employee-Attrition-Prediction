import pandas as pd
import joblib
import os

model = joblib.load("models/model.pkl")

def bulk_predict(df):

    df = df.copy()


    df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip().str.upper()


    X = df.drop(columns=["Employee_ID", "Attrition"], errors="ignore")


    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]


    df["Attrition_Predicted"] = predictions
    df["Probability"] = (probabilities * 100).round(2)

    df["Risk"] = df["Probability"].apply(
        lambda x: "High Risk" if x > 60 else
                  "Medium Risk" if x > 40 else
                  "Low Risk"
    )


    final_cols = [
        "Employee_ID",
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "JobSatisfaction",
        "Performance_Rating",
        "Work_Life_Balance",
        "Probability",
        "Risk"
    ]

    df = df[[col for col in final_cols if col in df.columns]]

    df.rename(columns={"Employee_ID": "EmployeeID"}, inplace=True)


    os.makedirs("result_data", exist_ok=True)


    high_df = df[df["Risk"] == "High Risk"]
    medium_df = df[df["Risk"] == "Medium Risk"]
    low_df = df[df["Risk"] == "Low Risk"]

    high_df = high_df.sort_values("Probability", ascending=False)
    medium_df = medium_df.sort_values("Probability", ascending=False)
    low_df = low_df.sort_values("Probability", ascending=False)


    full_path = "result_data/bulk_predictions.csv"
    high_path = "result_data/high_risk.csv"
    medium_path = "result_data/medium_risk.csv"
    low_path = "result_data/low_risk.csv"

    df.to_csv(full_path, index=False)
    high_df.to_csv(high_path, index=False)
    medium_df.to_csv(medium_path, index=False)
    low_df.to_csv(low_path, index=False)

    return full_path