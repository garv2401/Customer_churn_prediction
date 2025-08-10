# Customer Churn Prediction API 📊

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)
![Platform](https://img.shields.io/badge/Platform-Render-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An end-to-end machine learning project that predicts customer churn for a fictional telecom company. The trained model is deployed as a RESTful API on Render, allowing for real-time predictions.

---

## 🚀 Live API Demo

The API is live and can be accessed at the following URL. See the [API Usage](#-api-usage) section for instructions on how to interact with it.

**API Link:** [https://customer-churn-prediction-g1gl.onrender.com](https://customer-churn-prediction-g1gl.onrender.com/predict)

---

## ✨ Project Features

* **End-to-End ML Pipeline:** Covers the complete lifecycle from data cleaning and exploratory data analysis (EDA) to model deployment.
* **Class Imbalance Handling:** Implements the **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to address class imbalance.
* **Robust Model Evaluation:** Compares multiple classifiers (Random Forest, XGBoost) using 5-fold cross-validation to select the optimal model.
* **RESTful API:** The final model is served through a Flask API that accepts JSON input and returns churn predictions in real-time.
* **Cloud Deployment:** The entire application is container-ready and deployed on the Render cloud platform for public access.

---

## 🛠️ Technology Stack

* **Backend:** Python, Flask, Gunicorn
* **Data Science:** Pandas, Scikit-learn, XGBoost, Imbalanced-learn
* **Deployment:** Git, GitHub, Render

---

## ⚙️ Local Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/garv2401/Customer-Churn-Prediction.git](https://github.com/garv2401/Customer-Churn-Prediction.git)
    cd Customer-Churn-Prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    gunicorn app:app
    ```
    The application will be running on `http://127.0.0.1:8000`.

---

## 🚀 API Usage

You can interact with the API using tools like Postman or `curl`.

### Endpoint: `/predict`

* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

* **Request Body (raw JSON):**
    You must provide a JSON object with the customer's details.

    **Example 1 (Low Churn Risk):**
    ```json
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    ```

    **Example 2 (High Churn Risk):**
    ```json
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.45,
        "TotalCharges": 198.90
    }
    ```

* **Success Response (200 OK):**
    The API will return a JSON object with the prediction and its probability.
    ```json
    {
        "churn_prediction": "No",
        "churn_probability": 0.0763
    }
    ```

### Testing with `curl`

You can also test the endpoint from your terminal using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{ "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 9

