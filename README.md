🏠 House Price Prediction with Clustering

An end-to-end Machine Learning web application that predicts California house prices. This project uniquely combines Unsupervised Learning (KMeans) with Supervised Learning (Random Forest) to capture geographic price trends more effectively.

🔗 Live Demo: House Price Predictor App
🚀 Project Overview

Predicting real estate prices is complex because "location" is more than just coordinates—it's about neighborhoods. This application groups houses based on their Longitude and Latitude using KMeans clustering. These cluster labels are then used as a feature for the Random Forest Regressor, significantly improving the model's ability to understand local market trends.
Key Features

    🌍 Location-Based Intelligence: Uses KMeans to cluster properties by geographic coordinates.

    🤖 Hybrid ML Modeling: Combines clustering labels with structural features for prediction.

    📊 Interactive UI: A sleek Streamlit dashboard for real-time inference.

    💾 Efficient Inference: Pre-trained models and scalers are managed via Joblib for instant results.

    📈 Performance Metrics: Real-time evaluation using R² score.

🧠 Machine Learning Workflow

    Data Preprocessing: * Median imputation for missing values (specifically total_bedrooms).

    Feature scaling using StandardScaler for numerical consistency.

    Unsupervised Learning: * Applied KMeans Clustering to Latitude and Longitude to create a "Neighborhood" feature.

    Supervised Learning: * Trained a Random Forest Regressor using both the original housing features and the new cluster labels.

    Model Persistence: * Saved the trained model, scaler, and KMeans object using Joblib.

    Deployment: * Built the frontend with Streamlit and deployed to Streamlit Cloud.

🛠️ Tech Stack

    Language: Python

    Web Framework: Streamlit

    ML Libraries: Scikit-learn (Random Forest, KMeans, StandardScaler)

    Data Handling: Pandas, NumPy

    Visualization: Matplotlib

    Model Serialization: Joblib

📁 Dataset

The project utilizes the California Housing Dataset, which includes:

    Geographic: Longitude, Latitude

    Structural: Total rooms, total bedrooms, house age

    Demographic: Population, households, median income

    Target: Median house value

⚙️ Local Setup and Installation

To run this project locally, follow these steps:

    Clone the repository:
    Bash

git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

Create a virtual environment:
Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install dependencies:
Bash

pip install -r requirements.txt

Run the Streamlit app:
Bash

    streamlit run app.py

🎯 Use Case

This project serves as a practical example of Feature Engineering through clustering. By turning raw coordinates into meaningful geographic clusters, the model can account for the "location, location, location" rule of real estate more accurately than raw coordinates alone.
