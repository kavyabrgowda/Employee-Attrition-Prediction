from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import joblib
import os
from flask import send_file

from utils.bulk_predict import bulk_predict
from utils.metrics import dashboard_metrics

app = Flask(__name__)

# =========================
# LOAD MODEL (PIPELINE MODEL)
# =========================
model = joblib.load("models/model.pkl")


# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    metrics = dashboard_metrics("data/employee_data.csv")
    return render_template("dashboard.html", metrics=metrics)


# ---------------- SINGLE EMPLOYEE ----------------
@app.route("/single", methods=["GET", "POST"])
def single():

    if request.method == "POST":

        emp_id = request.form["Employee_ID"].strip().upper()

        df = pd.read_csv("data/employee_data.csv")
        df["Employee_ID"] = df["Employee_ID"].astype(str).str.strip().str.upper()

        row = df[df["Employee_ID"] == emp_id]

        if row.empty:
            return render_template("single_predict.html", error="Employee not found")

        data = row.iloc[0].to_dict()

        # DROP ID BEFORE PREDICTION (VERY IMPORTANT)
        input_df = pd.DataFrame([data]).drop(columns=["Employee_ID", "Attrition"], errors="ignore")

        prob = model.predict_proba(input_df)[0][1]

        # Risk classification
        if prob > 0.6:
            risk = "High Risk"
            color = "high"
        elif prob > 0.4:
            risk = "Medium Risk"
            color = "medium"
        else:
            risk = "Low Risk"
            color = "low"

        # =========================
        # DRIVER ANALYSIS (IMPROVED)
        # =========================
        drivers = []

        if data["JobSatisfaction"] <= 2:
            drivers.append("Low Job Satisfaction")

        if data["YearsSinceLastPromotion"] > 3:
            drivers.append("No Promotion in 3+ Years")

        if data["BusinessTravel"] == "Travel_Frequently":
            drivers.append("Frequent Travel Load")

        if data["MonthlyIncome"] < 40000:
            drivers.append("Below Market Salary")

        if data["Work_Life_Balance"] <= 2:
            drivers.append("Poor Work-Life Balance")

        recommendation = [
            "Schedule HR one-on-one discussion",
            "Offer career growth plan",
            "Improve engagement & recognition programs",
            "Review compensation structure"
        ]

        return render_template(
            "result.html",
            employee_id=emp_id,
            probability=round(prob * 100, 2),
            risk=risk,
            color=color,
            drivers=drivers,
            recommendation=recommendation,
            details=data
        )

    return render_template("single_predict.html")


# ---------------- BULK UPLOAD ----------------
@app.route("/bulk")
def bulk():
    return render_template("bulk_upload.html")


@app.route("/bulk_predict", methods=["POST"])
def bulk_upload():
    file = request.files["file"]
    df = pd.read_csv(file)

    result_file = bulk_predict(df)
    return send_file(result_file, as_attachment=True)


# ---------------- ANALYZE CSV ----------------
@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html", show_results=False)


@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():

    file = request.files.get("file")

    if not file:
        return redirect("/analyze")

    df = pd.read_csv(file)

    if "Risk" not in df.columns or "Probability" not in df.columns:
        return "Invalid File. Missing Risk or Probability column"

    # convert safely
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")

    # =========================
    # COUNTS
    # =========================
    high = (df["Risk"] == "High Risk").sum()
    medium = (df["Risk"] == "Medium Risk").sum()
    low = (df["Risk"] == "Low Risk").sum()

    total = len(df)
    high_percent = round((high / total) * 100, 2)

    # =========================
    # SORTED TOP 20 (IMPORTANT FIX)
    # =========================
    top_high = df[df["Risk"] == "High Risk"] \
        .sort_values("Probability", ascending=False).head(20)

    top_medium = df[df["Risk"] == "Medium Risk"] \
        .sort_values("Probability", ascending=False).head(20)

    top_low = df[df["Risk"] == "Low Risk"] \
        .sort_values("Probability", ascending=False).head(20)

    return render_template(
        "analyze.html",
        show_results=True,
        high=high,
        medium=medium,
        low=low,
        total=total,
        high_percent=high_percent,
        top_high=top_high.to_dict("records"),
        top_medium=top_medium.to_dict("records"),
        top_low=top_low.to_dict("records")
    )



# =========================
# DOWNLOAD HIGH RISK
# =========================
@app.route("/download_high")
def download_high():
    file_path = "result_data/high_risk.csv"
    return send_file(file_path, as_attachment=True)

# =========================
# DOWNLOAD MEDIUM RISK
# =========================
@app.route("/download_medium")
def download_medium():
    file_path = "result_data/medium_risk.csv"
    return send_file(file_path, as_attachment=True)

# =========================
# DOWNLOAD LOW RISK
# =========================
@app.route("/download_low")
def download_low():
    file_path = "result_data/low_risk.csv"
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)