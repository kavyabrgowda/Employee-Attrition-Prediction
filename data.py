import pandas as pd
import numpy as np

np.random.seed(42)
n = 1500

departments = ["Sales", "HR", "R&D", "Finance", "IT"]
job_roles = ["Manager", "Executive", "Analyst", "Engineer", "Clerk"]
business_travel = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
marital_status = ["Single", "Married", "Divorced"]

data = {
    "Age": np.random.randint(21, 60, n),
    "MaritalStatus": np.random.choice(marital_status, n),
    "DistanceFromHome": np.random.randint(1, 40, n),
    "Department": np.random.choice(departments, n),
    "JobRole": np.random.choice(job_roles, n),
    "BusinessTravel": np.random.choice(business_travel, n),
    "JobSatisfaction": np.random.randint(1, 5, n),
    "YearsAtCompany": np.random.randint(0, 25, n),
    "YearsSinceLastPromotion": np.random.randint(0, 10, n),
    "MonthlyIncome": np.random.randint(18000, 150000, n),
}

df = pd.DataFrame(data)

# Logical Attrition Pattern
df["Attrition"] = (
    (df["JobSatisfaction"] <= 2) &
    (df["MonthlyIncome"] < 40000) |
    (df["YearsSinceLastPromotion"] > 5) |
    (df["YearsAtCompany"] < 2)
).astype(int)

df.to_csv("data/employee_data.csv", index=False)

print("employee_data.csv generated successfully in data folder")