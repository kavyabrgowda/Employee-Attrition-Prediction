from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import joblib
from utils.preprocessing import preprocess_input
from utils.bulk_predict import bulk_predict
from utils.metrics import dashboard_metrics


app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route("/")
def dashboard():
    metrics = dashboard_metrics("data/employee_data.csv")
    return render_template("dashboard.html", metrics=metrics)

@app.route("/single")
def single():
    return render_template("single_predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = dict(request.form)
    processed = preprocess_input(input_data)

    probability = model.predict_proba(processed)[0][1]
    probability_percent = int(round(probability * 100))

    # Risk classification
    if probability > 0.6:
        risk = "High Risk"
        color = "high"
        recommendation = [
            "Schedule retention discussion",
            "Review compensation structure",
            "Discuss career growth opportunities"
        ]
    elif probability > 0.4:
        risk = "Medium Risk"
        color = "medium"
        recommendation = [
            "Monitor employee engagement",
            "Offer skill development program",
            "Check workload balance"
        ]
    else:
        risk = "Low Risk"
        color = "low"
        recommendation = [
            "Maintain engagement",
            "Recognize performance",
            "Continue career progression support"
        ]

    # Example key drivers (simple rule-based explanation)
    drivers = []

    if int(input_data["JobSatisfaction"]) <= 2:
        drivers.append("Low Job Satisfaction")

    if int(input_data["YearsSinceLastPromotion"]) > 3:
        drivers.append("No Promotion in 3+ Years")

    if input_data["BusinessTravel"] == "Travel_Frequently":
        drivers.append("Frequent Business Travel")

    if int(input_data["MonthlyIncome"]) < 40000:
        drivers.append("Below Market Salary")

    return render_template(
        "result.html",
        probability=probability_percent,
        risk=risk,
        color=color,
        drivers=drivers,
        recommendation=recommendation
    )

@app.route("/bulk")
def bulk():
    return render_template("bulk_upload.html")

@app.route("/bulk_predict", methods=["POST"])
def bulk_upload():
    file = request.files["file"]
    df = pd.read_csv(file)

    result_file = bulk_predict(df)
    return send_file(result_file, as_attachment=True)

@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html", show_results=False)


@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():

    file = request.files.get("file")

    if not file:
        return redirect("/analyze")

    df = pd.read_csv(file)

    if "Risk" not in df.columns:
        return "Invalid File. Please upload prediction CSV."

    high = (df["Risk"] == "High Risk").sum()
    medium = (df["Risk"] == "Medium Risk").sum()
    low = (df["Risk"] == "Low Risk").sum()

    total = len(df)
    high_percent = round((high / total) * 100, 2)

    # Top 20 High
    top_high = df[df["Risk"] == "High Risk"].head(20).copy()
    top_high.insert(0, "EmployeeID",
                    ["EMP" + str(i).zfill(4) for i in range(1, len(top_high)+1)])

    # Top 20 Medium
    top_medium = df[df["Risk"] == "Medium Risk"].head(20).copy()
    top_medium.insert(0, "EmployeeID",
                      ["EMP" + str(i).zfill(4) for i in range(1, len(top_medium)+1)])

    # Top 20 Low
    top_low = df[df["Risk"] == "Low Risk"].head(20).copy()
    top_low.insert(0, "EmployeeID",
                   ["EMP" + str(i).zfill(4) for i in range(1, len(top_low)+1)])

    avg_income_high = df[df["Risk"] == "High Risk"]["MonthlyIncome"].mean()

    return render_template(
        "analyze.html",
        show_results=True,
        high=high,
        medium=medium,
        low=low,
        total=total,
        high_percent=high_percent,
        avg_income_high=round(avg_income_high, 2) if not pd.isna(avg_income_high) else 0,
        top_high=top_high.to_dict(orient="records"),
        top_medium=top_medium.to_dict(orient="records"),
        top_low=top_low.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)