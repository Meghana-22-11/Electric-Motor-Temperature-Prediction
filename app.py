import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# load model and scaler
model = joblib.load("model.save")
trans = joblib.load("transform.save")


@app.route('/')
def home():
    return render_template('Manual_predict.html')


@app.route('/y_predict', methods=['POST'])
def y_predict():

    x_test = [[float(x) for x in request.form.values()]]

    x_test = trans.transform(x_test)

    prediction = model.predict(x_test)

    return render_template(
        'Manual_predict.html',
        prediction_text=f"Permanent Magnet surface temperature: {prediction[0]}"
    )


if __name__ == "__main__":
   app.run(debug=True, port=5002)