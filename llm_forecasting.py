# Import necessary libraries
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from flask import Flask, request, jsonify
import openai
import json
import io

# Initialize Flask app
app = Flask(__name__)

# Configuration for OpenAI API
OPENAI_API_KEY = None  # Replace with your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Add a route for the root URL
@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the LLM Forecasting API. Available endpoints: /train, /forecast, /llm-forecast'
    }), 200

# Function to fetch sample sales data from an online source
def fetch_sample_data():
    """
    Fetches sample sales data for demonstration purposes.
    In a real-world scenario, replace this with actual data sources.
    """
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(io.StringIO(response.text))
        return data
    else:
        raise Exception('Failed to fetch data')

# Function to preprocess the data
def preprocess_data(data):
    """
    Preprocesses the sales data by parsing dates and creating time-based features.
    """
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)
    data['Month_Num'] = data.index.month
    data['Year'] = data.index.year
    data['Lag_1'] = data['Passengers'].shift(1)
    data['Lag_2'] = data['Passengers'].shift(2)
    data['Rolling_Mean_3'] = data['Passengers'].rolling(window=3).mean()
    data.dropna(inplace=True)
    return data

# Function to train forecasting models
def train_models(data):
    """
    Trains a Random Forest model on the preprocessed data and saves the model.
    """
    features = ['Month_Num', 'Year', 'Lag_1', 'Lag_2', 'Rolling_Mean_3']
    target = 'Passengers'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error: {mae}")
    
    # Save the trained model
    joblib.dump(model, 'random_forest_model.joblib')
    return model, mae

# Function to generate forecast
def generate_forecast(model, last_known_data, periods=12):
    """
    Generates future forecasts using the trained model.
    """
    forecasts = []
    current_data = last_known_data.copy()

    for _ in range(periods):
        # Ensure current_data is a DataFrame with a single row
        if isinstance(current_data, pd.Series):
            current_data = current_data.to_frame().T

        X_pred = current_data[['Month_Num', 'Year', 'Lag_1', 'Lag_2', 'Rolling_Mean_3']].values.reshape(1, -1)
        pred = model.predict(X_pred)[0]
        forecasts.append(pred)

        # Update the current_data with the new prediction
        month_num = current_data['Month_Num'].iloc[0]
        year = current_data['Year'].iloc[0]

        new_entry = {
            'Month_Num': (month_num % 12) + 1,
            'Year': year + (1 if month_num == 12 else 0),
            'Lag_1': pred,
            'Lag_2': current_data['Lag_1'].iloc[0],
            'Rolling_Mean_3': (current_data['Lag_1'].iloc[0] + current_data['Lag_2'].iloc[0] + pred) / 3
        }
        current_data = pd.DataFrame([new_entry])

    return forecasts

# Endpoint to fetch and train data
@app.route('/train', methods=['POST'])
def train_endpoint():
    try:
        # Fetch and preprocess data
        data = fetch_sample_data()
        preprocessed_data = preprocess_data(data)
        
        # Train models
        model, mae = train_models(preprocessed_data)
        
        return jsonify({
            'message': 'Model trained successfully',
            'mean_absolute_error': mae
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to get forecasts
@app.route('/forecast', methods=['POST'])
def forecast_endpoint():
    try:
        # Load the trained model
        model = joblib.load('random_forest_model.joblib')
        
        # Fetch and preprocess data
        data = fetch_sample_data()
        preprocessed_data = preprocess_data(data)
        
        # Get the last known data point
        last_known = preprocessed_data.iloc[-1]
        
        # Generate forecasts
        forecasts = generate_forecast(model, last_known, periods=12)
        
        # Prepare future dates
        last_date = preprocessed_data.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=12, freq='M')
        
        forecast_df = pd.DataFrame({
            'Month': future_dates,
            'Predicted_Passengers': forecasts
        })
        
        return jsonify({
            'forecasts': forecast_df.to_dict(orient='records')
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for LLM Integration
@app.route('/llm-forecast', methods=['POST'])
def llm_forecast_endpoint():
    try:
        user_input = request.json.get('prompt')
        if not user_input:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Fetch and preprocess data
        data = fetch_sample_data()
        preprocessed_data = preprocess_data(data)
        
        # Load the trained model
        model = joblib.load('random_forest_model.joblib')
        
        # Generate forecast
        last_known = preprocessed_data.iloc[-1]
        forecasts = generate_forecast(model, last_known, periods=12)
        
        # Prepare future dates
        last_date = preprocessed_data.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=12, freq='M')
        
        # Convert future_dates to string format
        future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()
        
        forecast_df = pd.DataFrame({
            'Month': future_dates_str,
            'Predicted_Passengers': forecasts
        })
        
        # Prepare the message for the LLM
        forecast_summary = forecast_df.to_dict(orient='records')
        summary_text = json.dumps(forecast_summary, indent=2)
        
        # Generate actionable insights
        llm_prompt = f"""
        User Prompt: {user_input}
        
        Forecast Data:
        {summary_text}
        
        Provide actionable insights based on the above forecast data.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in business forecasting and data analysis."},
                {"role": "user", "content": llm_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        insights = response.choices[0].message['content'].strip()
        
        return jsonify({
            'forecasts': forecast_summary,
            'insights': insights
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    # Train the model initially
    try:
        data = fetch_sample_data()
        preprocessed_data = preprocess_data(data)
        model, mae = train_models(preprocessed_data)
        print("Model trained successfully with MAE:", mae)
    except Exception as e:
        print("Error during initial training:", e)
    
    # Run Flask app
    app.run(debug=True)
