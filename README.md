# Determining Air Quality  Using Image Processing

## Project Overview
This project is focused on predicting PM2.5 particulate matter concentration using image-based features along with other environmental factors. The system uses a Convolutional Neural Network (CNN) model to estimate air quality from images, combined with feature data such as place, AQI category, and concentration values.

## Features
- **Image Feature Extraction**: Uses dark channel, transmission estimation, sky color, and other image-based features to analyze air quality.
- **Machine Learning Model**: A pre-trained CNN model is used for predicting PM2.5 concentration.
- **Data Input**: Takes both image input and textual data (place, AQI, date, etc.) to generate predictions.
- **Flask Web Application**: Provides a simple web interface for users to upload images and receive predictions.

## Tools and Technologies Used
- **Backend**: Python, Flask
- **Image Processing**: OpenCV, NumPy
- **Machine Learning**: Keras (TensorFlow backend), Scikit-learn
- **Frontend**: HTML, CSS, JavaScript (using Jinja templates for rendering results)
- **Model Deployment**: Pre-trained CNN model for PM2.5 prediction
- **Database**: N/A
- **API**: Flask for routing and serving the web interface

