import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/employee_data.csv")

# Drop ID
df = df.drop(columns=["Employee_ID"], errors="ignore")

# =========================
# SPLIT FEATURES & TARGET
# =========================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# =========================
# COLUMN TYPES
# =========================
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# =========================
# PREPROCESSING PIPELINE
# =========================
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# =========================
# MODEL (IMPROVED)
# =========================
model = RandomForestClassifier(
    n_estimators=600,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# =========================
# PIPELINE
# =========================
clf = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# =========================
# SPLIT (IMPORTANT: STRATIFY)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# TRAIN
# =========================
clf.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = clf.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\n================ FINAL RESULTS ================\n")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(clf, "models/model.pkl")

print("\n✅ MODEL TRAINED & SAVED SUCCESSFULLY")