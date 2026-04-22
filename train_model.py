import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import os
os.makedirs("model", exist_ok=True)

import pickle
import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/flight_price_data.csv")

print("Initial shape:", df.shape)
# =========================
# 2. DATA CLEANING
# =========================
df.dropna(inplace=True)

# =========================
# 3. USE EXISTING FEATURES
# =========================

# Combine duration into minutes
df["Duration_mins"] = df["Duration_hours"] * 60 + df["Duration_min"]

# Drop original duration columns
df.drop(["Duration_hours", "Duration_min"], axis=1, inplace=True)

# =========================
# 4. HANDLE STOPS
# =========================
df["Total_Stops"] = df["Total_Stops"].map({
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
})

df["Total_Stops"] = df["Total_Stops"].fillna(0)

# =========================
# 5. DROP USELESS COLUMNS
# =========================
df.drop(["Date"], axis=1, inplace=True)

# =========================
# 6. DROP USELESS COLUMN
# =========================
if "Route" in df.columns:
    df.drop(["Route"], axis=1, inplace=True)

# =========================
# 7. ENCODING
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# 8. SPLIT DATA
# =========================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. TRAIN MODELS
# =========================
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# =========================
# 10. EVALUATION FUNCTION
# =========================
def evaluate(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)
evaluate("XGBoost", y_test, xgb_pred)

# =========================
# 11. SAVE MODEL
# =========================
pickle.dump(xgb, open("model/model.pkl", "wb"))
pickle.dump(X.columns, open("model/features.pkl", "wb"))

print("\nModel saved successfully!")