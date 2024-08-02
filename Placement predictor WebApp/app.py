from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the model
model = joblib.load('placement_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    prediction = model.predict(final_features)
    
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    
    return render_template('index.html', prediction_text='Placement Status: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
