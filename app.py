from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_features = [float(x) for x in request.form.values()]
    feature_array = np.array(input_features).reshape(1, -1)
    
    # Predict using the model
    prediction = model.predict(feature_array)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Days: {prediction:.2f}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
