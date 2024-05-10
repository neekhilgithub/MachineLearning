from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)

# Load the models and scaler
regressor_lr = joblib.load('regression_model.joblib')
regressor_xgb = joblib.load('xgboost_model.joblib')
regressor_rf = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        selected_model = request.form['model']  # User-selected model
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = int(request.form['sex'])
        smoker = int(request.form['smoker'])
        region_northwest = int(request.form['region_northwest'])
        region_southeast = int(request.form['region_southeast'])
        region_southwest = int(request.form['region_southwest'])

        # Feature scaling
        input_data = scaler.transform([[age, bmi, children, sex, smoker, region_northwest, region_southeast, region_southwest]])

        # Make predictions based on the selected model
        if selected_model == 'linear_regression':
            prediction = regressor_lr.predict(input_data)[0]
        elif selected_model == 'xgboost':
            prediction = regressor_xgb.predict(input_data)[0]
        elif selected_model == 'random_forest':
            prediction = regressor_rf.predict(input_data)[0]
        else:
            return render_template('index.html', prediction_text="Invalid model selection")

        # Render the 'result.html' template with the prediction results
        return render_template('result.html', model=selected_model, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)