import pickle
import pandas as pd

# =========================
# LOAD MODEL + FEATURES
# =========================
model = pickle.load(open("model/model.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

# =========================
# USER INPUT (EDIT THIS)
# =========================
input_data = {
    "Total_Stops": 1,
    "Month": 3,
    "Year": 2019,
    "Dep_hours": 10,
    "Dep_min": 30,
    "Arrival_hours": 14,
    "Arrival_min": 0,
    "Duration_mins": 240,

    # ONE-HOT ENCODED VALUES (example)
    "Airline_IndiGo": 1,
    "Airline_Air India": 0,
    "Source_Delhi": 1,
    "Destination_Cochin": 1
}

# =========================
# CONVERT TO DATAFRAME
# =========================
input_df = pd.DataFrame([input_data])

# Match training features
input_df = input_df.reindex(columns=features, fill_value=0)

# =========================
# PREDICT
# =========================
prediction = model.predict(input_df)

print(f"Predicted Flight Price: ₹{int(prediction[0])}")