import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

# --- Load the Model and Encoders ---
# Load the dictionary containing the model and feature names
model_data = pickle.load(open('customer_churn_model.pkl', 'rb'))
model = model_data['model']
model_feature_names = model_data['feature_names']

# Load the dictionary of encoders
encoders = pickle.load(open('encoders.pkl', 'rb'))


# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    json_data = request.get_json(force=True)
    
    # Convert the JSON data into a pandas DataFrame
    input_df = pd.DataFrame(json_data, index=[0])

    # --- Preprocessing the input data ---
    # Apply the saved label encoders to the categorical columns
    for column, encoder in encoders.items():
        if column in input_df.columns:
            # Use a lambda function to handle unseen labels gracefully
            # It maps unseen values to -1, which you may need to handle
            # or ensure your input data is clean.
            input_df[column] = input_df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
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
    probability = float(prediction_proba[0][1]) # Probability of churn (class 1)

    return jsonify({
        'churn_prediction': churn_status,
        'churn_probability': round(probability, 4)
    })

if __name__ == '__main__':
    # The port is set to what Heroku expects
    app.run(host='0.0.0.0', port=5000)