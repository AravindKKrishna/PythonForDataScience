from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

data = load_breast_cancer()
feature_names = list(data.feature_names)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[feature]) for feature in feature_names]

    input_data = np.array([values])

    scaled = scaler.transform(input_data)
    reduced = pca.transform(scaled)
    prediction = model.predict(reduced)

    result = "Malignant" if prediction[0] == 0 else "Benign"

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction_text=f"Prediction: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)
