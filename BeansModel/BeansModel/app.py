from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

df = pd.read_csv(r"D:\Data_Science_and_Machine_Learning\BeansModel\BeansModel\Dry_Bean.csv")

feature_names = list(df.drop("Class", axis=1).columns)

model = pickle.load(open("bestModel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label = pickle.load(open("label_encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[feature]) for feature in feature_names]

    input_data = np.array([values])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)

    decoded = label.inverse_transform(prediction)[0]

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction_text=f"Predicted Bean Type: {decoded}"
    )


if __name__ == "__main__":
    app.run(debug=True)
