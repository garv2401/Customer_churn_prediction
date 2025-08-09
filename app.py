import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

# --- Load the Model and Encoders Separately ---
try:
    with open('churn_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('encoders.pkl', 'rb') as f_encoders:
        encoders = pickle.load(f_encoders)

except FileNotFoundError:
    # This will help you debug if files are missing in the deployment
    model = None
    encoders = None

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None:
        return jsonify({'error': 'Model or encoders not loaded. Check server logs.'}), 500

    # Get the JSON data from the request
    json_data = request.get_json(force=True)
    
    # Convert the JSON data into a pandas DataFrame
    input_df = pd.DataFrame(json_data, index=[0])

    # --- Preprocessing the input data ---
    # Apply the saved label encoders
    for column, encoder in encoders.items():
        if column in input_df.columns:
            # Use a lambda to handle unseen values during prediction
            input_df[column] = input_df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
    # Get the feature names the model was trained on
    model_feature_names = model.feature_names_in_

    # Ensure all required feature columns are present, fill missing with 0
    for col in model_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the model's training order
    input_df = input_df[model_feature_names]

    # --- Make Prediction ---
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # --- Format the Response ---
    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    probability = float(prediction_proba[0][1])

    return jsonify({
        'churn_prediction': churn_status,
        'churn_probability': round(probability, 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)