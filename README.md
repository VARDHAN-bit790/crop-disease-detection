# -FarmTech
# Crop Disease Prediction Web App

This is a web application built using Flask and TensorFlow for predicting crop diseases. The app allows users to upload images of crop leaves, which are then analyzed using a pre-trained deep learning model to detect diseases.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Model Details](#model-details)
- [How to Use](#how-to-use)
- [File Structure](#file-structure)
- [License](#license)

## Features

- **Disease Prediction**: Upload images of crop leaves, and the app will predict the disease based on the uploaded image.
- **Model**: Uses a MobileNetV2-based Convolutional Neural Network (CNN) for disease classification.
- **Frontend**: Clean and responsive frontend built with HTML and Bootstrap.
- **Backend**: Flask-based backend for handling image upload, model prediction, and result display.

## Technologies Used

- **Flask**: Web framework for building the backend of the app.
- **TensorFlow**: Deep learning framework used to train the model and make predictions.
- **Keras**: For building and training the model.
- **Bootstrap**: For responsive design of the frontend.
- **NumPy**: For numerical operations like processing images.
- **Matplotlib**: For displaying predictions and confusion matrix.
- **Werkzeug**: For secure file handling.

## Setup and Installation

### Prerequisites

Ensure that you have the following installed:

- Python 3.x
- pip

### Install Dependencies

Clone the repository and install the required packages.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt

