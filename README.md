# Fraud Detection Autoencoder

## Description
This project implements an **Autoencoder** for detecting fraudulent transactions in banking data. The model is trained on normal transactions and identifies anomalies, which are potential frauds. The approach uses **unsupervised learning**, making it suitable for real-world applications with limited labeled data. 

The project also demonstrates how to deploy the model using **FastAPI** to create a RESTful API and **Docker** for containerization, allowing for easy deployment and scalability.

## Features
- **Fraud Detection Model**: Based on an Autoencoder, which detects anomalies in the data.
- **FastAPI**: Used to build a simple and efficient REST API for the model.
- **Docker**: Docker is used to containerize the application, ensuring easy deployment and scalability.

## Technologies
- **TensorFlow**: Deep learning framework used to build the Autoencoder.
- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **Docker**: For containerization and deployment.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: For data preprocessing and model evaluation.

## Installation

### Prerequisites
- Python 3.12
- Docker (if you want to use the containerized version)
