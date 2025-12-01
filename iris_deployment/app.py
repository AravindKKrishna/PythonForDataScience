from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def modelpredict():
    data = [float(x) for x in request.form.values()]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    if prediction==0:
        result='setosa'
        
    elif prediction==1:
        result='versicolor'
    elif prediction==2:
        result='verginica'
    
    return render_template('index.html', predict_text=f"The predicted species is {result}")

if __name__ == "__main__":
    app.run(debug=True)
