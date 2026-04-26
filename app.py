from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    data = {
        "Total_Stops": int(form["stops"]),
        "Month": int(form["month"]),
        "Year": int(form["year"]),
        "Dep_hours": int(form["dep_hour"]),
        "Dep_min": int(form["dep_min"]),
        "Arrival_hours": int(form["arr_hour"]),
        "Arrival_min": int(form["arr_min"]),
        "Duration_mins": int(form["duration"])
    }

    # =========================
    # 🟢 FEATURE ENGINEERING (MATCH TRAINING)
    # =========================

    # Time difference
    time_diff = (
        (data["Arrival_hours"] * 60 + data["Arrival_min"]) -
        (data["Dep_hours"] * 60 + data["Dep_min"])
    )
    if time_diff < 0:
        time_diff += 1440
    data["Time_diff"] = time_diff

    # Peak hour
    dep_hour = data["Dep_hours"]
    data["Is_peak_dep"] = 1 if (6 <= dep_hour <= 10 or 17 <= dep_hour <= 21) else 0

    # Duration category (one-hot)
    dur = data["Duration_mins"]
    data["Duration_cat_1"] = 0
    data["Duration_cat_2"] = 0
    data["Duration_cat_3"] = 0

    if 120 < dur <= 300:
        data["Duration_cat_1"] = 1
    elif 300 < dur <= 600:
        data["Duration_cat_2"] = 1
    elif dur > 600:
        data["Duration_cat_3"] = 1

    # =========================
    # 🟢 NEW FEATURES (IMPORTANT)
    # =========================

    airline = form.get("airline")
    source = form.get("source")
    destination = form.get("destination")

    # Route
    route = f"{source}_{destination}"

    # Time block
    if 5 <= dep_hour < 12:
        data["Dep_time_block"] = 0
    elif 12 <= dep_hour < 17:
        data["Dep_time_block"] = 1
    elif 17 <= dep_hour < 22:
        data["Dep_time_block"] = 2
    else:
        data["Dep_time_block"] = 3

    # Interaction
    data["Stops_x_Duration"] = data["Total_Stops"] * data["Duration_mins"]

    # =========================
    # 🟢 DYNAMIC ONE-HOT ENCODING
    # =========================

    # Airlines
    for col in features:
        if col.startswith("Airline_"):
            data[col] = 0

    if f"Airline_{airline}" in features:
        data[f"Airline_{airline}"] = 1

    # Source
    for col in features:
        if col.startswith("Source_"):
            data[col] = 0

    if f"Source_{source}" in features:
        data[f"Source_{source}"] = 1

    # Destination
    for col in features:
        if col.startswith("Destination_"):
            data[col] = 0

    if f"Destination_{destination}" in features:
        data[f"Destination_{destination}"] = 1

    # Route
    for col in features:
        if col.startswith("Route_"):
            data[col] = 0

    if f"Route_{route}" in features:
        data[f"Route_{route}"] = 1

    # =========================
    # 🟢 FINAL DATAFRAME
    # =========================
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)

    return render_template(
        "index.html",
        prediction=int(prediction[0]),
        form_data=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)