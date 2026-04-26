def predict():
    data = {
        "Total_Stops": int(request.form["stops"]),
        "Month": int(request.form["month"]),
        "Year": int(request.form["year"]),
        "Dep_hours": int(request.form["dep_hour"]),
        "Dep_min": int(request.form["dep_min"]),
        "Arrival_hours": int(request.form["arr_hour"]),
        "Arrival_min": int(request.form["arr_min"]),
        "Duration_mins": int(request.form["duration"])
    }

    # ADD MISSING ML FEATURES

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

    # Duration category (one-hot like training)
    dur = data["Duration_mins"]
    if dur <= 120:
        # base case → all 0
        pass
    elif dur <= 300:
        data["Duration_cat_1"] = 1
    elif dur <= 600:
        data["Duration_cat_2"] = 1
    else:
        data["Duration_cat_3"] = 1

    # GET FORM VALUES
    airline = request.form.get("airline")
    source = request.form.get("source")
    destination = request.form.get("destination")

    # AIRLINE MAPPING
    data["Airline_IndiGo"] = 1 if airline == "IndiGo" else 0
    data["Airline_Air India"] = 1 if airline == "Air India" else 0
    data["Airline_SpiceJet"] = 1 if airline == "SpiceJet" else 0
    data["Airline_Vistara"] = 1 if airline == "Vistara" else 0

    # SOURCE MAPPING
    data["Source_Delhi"] = 1 if source == "Delhi" else 0
    data["Source_Mumbai"] = 1 if source == "Mumbai" else 0
    data["Source_Kolkata"] = 1 if source == "Kolkata" else 0
    data["Source_Chennai"] = 1 if source == "Chennai" else 0
    data["Source_Bangalore"] = 1 if source == "Bangalore" else 0

    # DESTINATION MAPPING
    data["Destination_Cochin"] = 1 if destination == "Cochin" else 0
    data["Destination_Delhi"] = 1 if destination == "Delhi" else 0
    data["Destination_Hyderabad"] = 1 if destination == "Hyderabad" else 0
    data["Destination_Kolkata"] = 1 if destination == "Kolkata" else 0

    # FINAL DATAFRAME
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)

    return render_template(
        "index.html",
        prediction=int(prediction[0]),
        form_data=request.form
    )