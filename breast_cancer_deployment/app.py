from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])
    prediction = model.predict(final)

    output = "Malignant (Cancer Present)" if prediction[0] == 0 else "Benign (No Cancer)"
    return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
