🏠 House Price Prediction with Clustering

A comprehensive Machine Learning web application built with Streamlit that predicts house prices. By integrating geographical clustering with regression analysis, this tool provides more localized and accurate real estate valuations.
📖 Project Overview

Real estate value is heavily dependent on location. This application uses a hybrid approach: it first groups properties into geographical neighborhoods using KMeans Clustering (based on Latitude and Longitude) and then applies a Random Forest Regressor to predict the final price. This method allows the model to "understand" specific regional price trends.
✨ Key Features

    🌍 Location-Based Clustering: Automatically identifies geographical clusters to improve prediction accuracy.

    🤖 Hybrid ML Modeling: Combines Unsupervised (KMeans) and Supervised (Random Forest) learning.

    📈 Real-Time Inference: A user-friendly Streamlit interface for entering property details and getting instant price estimates.

    💾 Efficient Processing: Pre-trained models and scalers are loaded via Joblib to ensure fast response times without retraining.

    📊 Data Insights: Uses the California Housing Dataset to provide realistic, data-driven estimations.

💻 Tech Stack

    Frontend: Streamlit (Interactive Web Interface)

    Machine Learning: Scikit-Learn (Random Forest & KMeans)

    Data Processing: Pandas, NumPy

    Model Serialization: Joblib

    Visualization: Matplotlib, Seaborn

⚙️ Machine Learning Pipeline

    Data Preprocessing: Handling missing values via median imputation and feature scaling using StandardScaler.

    Feature Engineering: Applying KMeans Clustering to Latitude/Longitude to create a new "Cluster ID" feature.

    Model Training: Utilizing a Random Forest Regressor to handle the non-linear complexities of housing data.

    Serialization: Saving the pipeline (Scaler, KMeans, and Random Forest) for production use.

🚀 Local Installation & Setup

To run this application on your local machine, follow these steps:

    Clone the Repository:
    Bash

git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction

Set Up a Virtual Environment:
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
Bash

pip install -r requirements.txt

Launch the App:
Bash

    streamlit run app.py

📂 Project Structure
Plaintext

├── housing_data.csv     # California Housing Dataset
├── app.py               # Main Streamlit application logic
├── model_trainer.py     # Script used for training and saving models
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

🎯 Use Case

This project demonstrates the power of Hybrid Machine Learning. By turning raw coordinates into meaningful clusters, the model mimics how human real estate experts evaluate property based on specific neighborhood characteristics, leading to a more robust and interpretable prediction system
