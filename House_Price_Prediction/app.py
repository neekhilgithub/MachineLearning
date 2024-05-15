from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        input_features = np.array(features).reshape(1, -1)
        prediction = model.predict(input_features)
        return render_template('result.html', prediction=f'Predicted Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)