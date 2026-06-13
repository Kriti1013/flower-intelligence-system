from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/petal_length_model.pkl")
scaler = joblib.load("models/petal_length_scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_width = float(request.form["petal_width"])

    features = np.array([[sepal_length, sepal_width, petal_width]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Petal Length: {prediction[0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)