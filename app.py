from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

petal_model = joblib.load("models/petal_length_model.pkl")
petal_scaler = joblib.load("models/petal_length_scaler.pkl")

species_model = joblib.load("models/species_model.pkl")
species_scaler = joblib.load("models/species_scaler.pkl")
label_encoder = joblib.load("models/le_species.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/petal")
def petal():
    return render_template("petal.html")


@app.route("/species")
def species():
    return render_template("species.html")


@app.route("/predict_petal", methods=["POST"])
def predict_petal():

    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pw = float(request.form["petal_width"])

    data = np.array([[sl, sw, pw]])

    scaled = petal_scaler.transform(data)

    prediction = petal_model.predict(scaled)[0]

    return render_template(
        "petal.html",
        prediction_text=f"Predicted Petal Length: {prediction:.2f}"
    )


@app.route("/predict_species", methods=["POST"])
def predict_species():

    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    data = np.array([[sl, sw, pl, pw]])

    scaled = species_scaler.transform(data)

    pred = species_model.predict(scaled)

    flower = label_encoder.inverse_transform(pred)[0]

    return render_template(
        "species.html",
        prediction_text=f"Predicted Species: {flower}"
    )


if __name__ == "__main__":
    app.run(debug=True)
