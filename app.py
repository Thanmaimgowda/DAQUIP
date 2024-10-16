from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras.optimizers import Adam
from keras import metrics

# Initialize Flask application
app = Flask(__name__)

# Load the model and other necessary objects
model = None
scaler = None
encoder_place = None
encoder_aqi = None

# Custom metrics (if any)
def mse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

# Load the model and preprocessors
def load_model_and_preprocessors():
    global model, scaler, encoder_place, encoder_aqi

    # Load the model with custom objects
    model = load_model(
        '/Users/thanmai/Desktop/DAQUIP/models/cnn_model.h5',
        custom_objects={'mse': mse}
    )

    # Compile the model (ensure to specify loss and metrics)
    optimizer = Adam()  # Example optimizer, you can choose your own
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    # Load the scaler
    with open('/Users/thanmai/Desktop/DAQUIP/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the encoders
    with open('/Users/thanmai/Desktop/DAQUIP/models/encoder_place.pkl', 'rb') as f:
        encoder_place = pickle.load(f)

    with open('/Users/thanmai/Desktop/DAQUIP/models/encoder_aqi.pkl', 'rb') as f:
        encoder_aqi = pickle.load(f)

# Route to handle index/home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the model is loaded
    if model is None:
        load_model_and_preprocessors()

    # Get inputs from the form
    image_file = request.files['image']
    place = request.form['place']
    aqi_category = request.form['aqi_category']
    nowcast_conc = float(request.form['nowcast_conc'])
    raw_conc = float(request.form['raw_conc'])
    date = request.form['date']
    time = request.form['time']

    # Save the image temporarily and extract features
    image_path = save_image_and_return_path(image_file)

    try:
        # Extract image features
        image_features = extract_image_features(image_path)

        # Prepare other features
        date = datetime.strptime(date, '%Y-%m-%d')
        hour = date.hour
        day_of_week = date.weekday()  # Monday is 0 and Sunday is 6
        month = date.month

        # One-hot encode the place and AQI category
        place_encoded = encoder_place.transform([[place]]).toarray()
        aqi_encoded = encoder_aqi.transform([[aqi_category]]).toarray()

        # Combine all features
        features = np.hstack((
            image_features,
            place_encoded,
            aqi_encoded,
            [[nowcast_conc, raw_conc, hour, day_of_week, month]]
        ))

        # Normalize the features
        features = scaler.transform(features)

        # Make prediction
        pm25_prediction = model.predict(features)
        pm25_prediction = pm25_prediction.flatten()[0]  # Extract the prediction from the array

        return render_template('result.html', prediction=pm25_prediction)

    except Exception as e:
        error_message = str(e)
        return render_template('error.html', error_message=error_message)

# Function to save uploaded image and return its path
def save_image_and_return_path(image_file):
    # Ensure the 'uploads' folder exists
    uploads_dir = os.path.join(app.instance_path, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # Save the image to the uploads folder
    image_path = os.path.join(uploads_dir, image_file.filename)
    image_file.save(image_path)

    return image_path

def extract_dark_channel(image):
    min_channel = np.min(image, axis=2)
    return min_channel

def estimate_transmission(dark_channel, omega=0.95):
    transmission = 1 - omega * dark_channel
    return transmission

def get_sky_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    sky_color = np.mean(lab, axis=(0, 1))
    return sky_color

def power_spectrum_slope(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    slope = np.mean(magnitude_spectrum)
    return slope

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.sqrt(np.mean((gray - np.mean(gray))**2))
    return contrast

def calculate_normalized_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    normalized_saturation = (saturation - np.min(saturation)) / (np.max(saturation) - np.min(saturation))
    histogram = np.histogram(normalized_saturation, bins=10, range=(0, 1))[0]
    return histogram

# Define the function for feature extraction from images
def extract_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image: {image_path}")
        
        dark_channel = extract_dark_channel(image)
        transmission = estimate_transmission(dark_channel)
        sky_color = get_sky_color(image)
        slope = power_spectrum_slope(image)
        contrast = calculate_contrast(image)
        normalized_saturation = calculate_normalized_saturation(image)

        features = [
            float(np.mean(dark_channel)),
            float(np.mean(transmission)),
            float(np.mean(sky_color)),
            float(slope),
            float(contrast),
            *normalized_saturation.tolist()
        ]
        return features
    except Exception as e:
        print(e)
        return [np.nan] * 7

# Entry point for the application
if __name__ == '__main__':
    load_model_and_preprocessors()
    app.run(debug=True)
