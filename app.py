import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Create a Flask app
app = Flask(__name__)

# --- Load the Model, Encoders, and Feature Names ---
try:
    with open('churn_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('encoders.pkl', 'rb') as f_encoders:
        encoders = pickle.load(f_encoders)

    with open('feature_names.pkl', 'rb') as f_features:
        model_feature_names = pickle.load(f_features)

except Exception as e:
    # Log the error for debugging in Render
    print(f"Error loading files: {e}")
    model, encoders, model_feature_names = None, None, None


# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or encoders is None or model_feature_names is None:
        return jsonify({'error': 'Model or supporting files not loaded. Check server logs.'}), 500

    json_data = request.get_json(force=True)
    input_df = pd.DataFrame(json_data, index=[0])

    # --- Preprocessing ---
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = input_df[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    
    for col in model_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_feature_names]

    # --- Prediction ---
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    probability = float(prediction_proba[0][1])

    return jsonify({
        'churn_prediction': churn_status,
        'churn_probability': round(probability, 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)