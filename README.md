🏠 House Price Prediction with Clustering

This project is an end-to-end Machine Learning web application built using Streamlit, which predicts house prices by combining unsupervised learning (KMeans clustering) and supervised learning (Random Forest regression).

The application groups houses based on geographical location (longitude and latitude) using KMeans clustering, then uses the cluster labels along with housing features to improve the accuracy of house price predictions.
Trained models and scalers are saved and loaded using Joblib for efficient reuse and faster inference.

🚀 Features

📊 Interactive data exploration with Streamlit

🌍 Location-based clustering using KMeans

🤖 House price prediction using Random Forest Regressor

💾 Model persistence using Joblib

📈 Model evaluation using R² score

🏡 Real-time house price prediction via user inputs

🧠 Machine Learning Workflow

Data loading and preprocessing

Handling missing values using median imputation

Feature scaling with StandardScaler

Location-based clustering using KMeans

Model training with Random Forest Regression

Saving & loading models using Joblib

Real-time prediction through Streamlit UI

🛠️ Tech Stack

Python

Streamlit

Pandas, NumPy

Matplotlib

Scikit-learn

Joblib

📁 Dataset

The project uses the California Housing Dataset, which includes:

Housing median age

Total rooms and bedrooms

Population and households

Median income

Geographic coordinates (longitude & latitude)

🌐 Deployment

The application is deployed on Streamlit Cloud, enabling users to access the model directly from a web browser without local installation.

🎯 Use Case

This project demonstrates:

The power of combining clustering with regression

How to build a full ML pipeline

How to deploy a production-ready ML application
