from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
models = pickle.load(open("model/model.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

xgb = models["xgb"]
lr = models["lr"]

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

    # FEATURE ENGINEERING (MATCH TRAINING)

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

    # BASIC INPUTS
    airline = form.get("airline")
    source = form.get("source")
    destination = form.get("destination")

    # ONE-HOT ENCODING

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

    # FINAL DATAFRAME
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    # ENSEMBLE PREDICTION
    xgb_pred = xgb.predict(df)
    lr_pred = lr.predict(df)
    final_pred = 0.85 * xgb_pred + 0.15 * lr_pred

    return render_template(
        "index.html",
        prediction=int(final_pred[0]),
        form_data=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)