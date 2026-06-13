from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ==========================
# Load Models
# ==========================

species_model = joblib.load("models/species_model.pkl")
petal_length_model = joblib.load("models/petal_length_model.pkl")

species_scaler = joblib.load("models/species_scaler.pkl")
petal_length_scaler = joblib.load("models/petal_length_scaler.pkl")

le_species = joblib.load("models/le_species.pkl")

# ==========================
# Pages
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/species")
def species():
    return render_template("species.html")


@app.route("/petal_length")
def petal_length():
    return render_template("petal_length.html")


# ==========================
# Species Prediction
# ==========================

@app.route("/predict_species", methods=["POST"])
def predict_species():

    try:

        data = request.get_json()

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        features = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        features = species_scaler.transform(features)

        prediction = species_model.predict(features)

        species_name = le_species.inverse_transform(prediction)[0]

        return jsonify({
            "predicted_species": species_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# ==========================
# Petal Length Prediction
# ==========================

@app.route("/predict_petal_length", methods=["POST"])
def predict_petal_length():

    try:

        data = request.get_json()

        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_width = float(data["petal_width"])

        features = np.array([
            [
                sepal_length,
                sepal_width,
                petal_width
            ]
        ])

        features = petal_length_scaler.transform(features)

        prediction = petal_length_model.predict(features)

        return jsonify({
            "predicted_petal_length": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# ==========================
# Run App
# ==========================

if __name__ == "__main__":
    app.run(debug=True)