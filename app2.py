import gradio as gr
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from datetime import datetime

# Load the model, scaler, and encoders
model = load_model('/Users/thanmai/Desktop/DAQUIP/models/cnn_model.h5')

with open('/Users/thanmai/Desktop/DAQUIP/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/Users/thanmai/Desktop/DAQUIP/models/encoder_place.pkl', 'rb') as f:
    encoder_place = pickle.load(f)

with open('/Users/thanmai/Desktop/DAQUIP/models/encoder_aqi.pkl', 'rb') as f:
    encoder_aqi = pickle.load(f)

# Define a function to process inputs and make predictions
def predict_pm25(image_no, date, time, place, nowcastconc, aqi_category, raw_conc, conc_unit):
    # Convert inputs to appropriate formats
    date = datetime.strptime(date, '%Y-%m-%d')
    time = datetime.strptime(time, '%H:%M').time()
    hour = time.hour
    day_of_week = date.weekday()
    month = date.month

    # One-hot encode categorical features
    encoded_place = encoder_place.transform([[place]])
    encoded_aqi_category = encoder_aqi.transform([[aqi_category]])

    # Load and process image features (assuming you have a way to load and process the image features)
    # Placeholder for image features
    image_features = np.zeros((1, 7))  # Replace with actual feature extraction

    # Combine all features
    features = np.hstack((image_features, encoded_place, encoded_aqi_category, [[nowcastconc, raw_conc, hour, day_of_week, month]]))

    # Normalize/standardize the features
    features = scaler.transform(features)

    # Make prediction
    pm25_prediction = model.predict(features)
    return float(pm25_prediction[0])

# Create Gradio interface
inputs = [
    gr.Textbox(label='Image no.'),
    gr.Textbox(label='Date (YYYY-MM-DD)'),
    gr.Textbox(label='Time (HH:MM)'),
    gr.Textbox(label='Place'),
    gr.Textbox(label='NowCastConc.'),
    gr.Textbox(label='AQI Category'),
    gr.Textbox(label='Raw Conc.'),
    gr.Textbox(label='Conc. Unit')
]

outputs = gr.Textbox(label='Predicted PM2.5')

gr.Interface(fn=predict_pm25, inputs=inputs, outputs=outputs, title="PM2.5 Prediction", description="Enter the required values to predict PM2.5 levels").launch()
