import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

import os
os.makedirs("model", exist_ok=True)

import pickle
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data/flight_price_data.csv")

print("Initial shape:", df.shape)

# Basic cleaning
df.dropna(inplace=True)

# Feature engineering

# Convert duration into total minutes
df["Duration_mins"] = df["Duration_hours"] * 60 + df["Duration_min"]
df.drop(["Duration_hours", "Duration_min"], axis=1, inplace=True)

# Time difference
df["Time_diff"] = (
    (df["Arrival_hours"] * 60 + df["Arrival_min"]) -
    (df["Dep_hours"] * 60 + df["Dep_min"])
)
df["Time_diff"] = df["Time_diff"].apply(lambda x: x if x > 0 else x + 1440)

# Peak hour
df["Is_peak_dep"] = df["Dep_hours"].apply(
    lambda x: 1 if (6 <= x <= 10 or 17 <= x <= 21) else 0
)

# Duration category
df["Duration_cat"] = pd.cut(
    df["Duration_mins"],
    bins=[0, 120, 300, 600, 2000],
    labels=[0, 1, 2, 3]
)

# Stops
df["Total_Stops"] = df["Total_Stops"].map({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
})
df["Total_Stops"] = df["Total_Stops"].fillna(0)

# Drop columns
df.drop(["Date"], axis=1, inplace=True)

if "Route" in df.columns:
    df.drop(["Route"], axis=1, inplace=True)

# Encoding
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train models
# =========================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# =========================
# ENSEMBLE (XGB + LR)
# =========================
ensemble_pred = (0.85 * xgb_pred + 0.15 * lr_pred)

# =========================
# STACKING MODEL
# =========================
stack = StackingRegressor(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lr', lr)
    ],
    final_estimator=LinearRegression()
)

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

# =========================
# Evaluation
# =========================
def evaluate(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("XGBoost", y_test, xgb_pred)
evaluate("Ensemble (XGB + LR)", y_test, ensemble_pred)
evaluate("Stacking Model", y_test, stack_pred)

# =========================
# Save BEST model
# =========================
scores = {
    "xgb": r2_score(y_test, xgb_pred),
    "ensemble": r2_score(y_test, ensemble_pred),
    "stack": r2_score(y_test, stack_pred)
}

best = max(scores, key=scores.get)

if best == "stack":
    pickle.dump(stack, open("model/model.pkl", "wb"))
elif best == "ensemble":
    # save both models for ensemble
    pickle.dump({"xgb": xgb, "lr": lr}, open("model/model.pkl", "wb"))
else:
    pickle.dump(xgb, open("model/model.pkl", "wb"))

pickle.dump(X.columns, open("model/features.pkl", "wb"))

print("\nBest model:", best)
print("Model saved successfully!")