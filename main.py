
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("data/employee_data.csv")

# Drop ID
df = df.drop(columns=["Employee_ID"], errors="ignore")

print("\n================ DATA OVERVIEW ================")
print(df.info())

print("\n================ SUMMARY ================")
print(df.describe())


sns.set(style="whitegrid")

# Attrition Distribution
plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Attrition Distribution")
plt.show()

# Numerical Features Distribution
for col in ["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction"]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()

# Boxplots vs Attrition
for col in ["MonthlyIncome", "YearsAtCompany", "JobSatisfaction"]:
    plt.figure()
    sns.boxplot(x="Attrition", y=col, data=df)
    plt.title(f"{col} vs Attrition")
    plt.show()

# Categorical vs Attrition
for col in ["Department", "JobRole", "BusinessTravel"]:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue="Attrition", data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Attrition")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()



X = df.drop("Attrition", axis=1)
y = df["Attrition"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)


model = LogisticRegression(max_iter=1000)


clf = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]


print("\n================ FINAL RESULTS ================\n")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


feature_names = clf.named_steps["preprocess"].get_feature_names_out()
coefficients = clf.named_steps["model"].coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": coefficients
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
plt.title("Top 10 Important Features")
plt.show()


joblib.dump(clf, "models/model.pkl")

print("\nLOGISTIC REGRESSION MODEL TRAINED & SAVED SUCCESSFULLY")