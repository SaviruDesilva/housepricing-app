🏠 House Price Prediction with Clustering

An end-to-end Machine Learning web application that predicts house prices by combining Unsupervised Learning (KMeans Clustering) and Supervised Learning (Random Forest Regression).

By grouping houses based on geographic coordinates (Latitude/Longitude), the model captures location-based trends more effectively, leading to more accurate price estimations.
🚀 Live Demo

Check out the web app here: House Price Predictor
✨ Features

    Hybrid ML Approach: Integrates KMeans labels into a Random Forest Regressor to improve accuracy.

    Interactive UI: Built with Streamlit for real-time price predictions based on user input.

    Data-Driven Insights: Uses the California Housing Dataset to provide realistic estimations.

    Model Persistence: Models and scalers are serialized using joblib for fast, efficient inference without retraining.

    Automated Preprocessing: Handles missing values via median imputation and feature scaling using StandardScaler.

🧠 Machine Learning Workflow

    Data Preprocessing: Cleaning the California Housing dataset and handling missing values.

    Feature Engineering: * KMeans Clustering: Identifying geographic clusters based on longitude and latitude.

        Scaling: Standardizing numerical features for better model performance.

    Model Training: Training a Random Forest Regressor on both raw features and the generated cluster labels.

    Evaluation: Validating performance using the R² Score.

    Deployment: Exporting the trained model (.pkl) and serving it via a Streamlit web interface.

🛠️ Tech Stack

    Frontend: Streamlit

    Language: Python

    Data Analysis: Pandas, NumPy

    Machine Learning: Scikit-learn

    Model Storage: Joblib

    Visualization: Matplotlib, Seaborn

📁 Dataset

The project utilizes the California Housing Dataset, which contains:

    MedInc: Median income in block group

    HouseAge: Median house age in block group

    AveRooms: Average number of rooms per household

    AveBedrms: Average number of bedrooms per household

    Population: Block group population

    AveOccup: Average number of household members

    Latitude & Longitude: Geographic location

⚙️ Local Installation

    Clone the repository:
    Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

Create a virtual environment:
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
Bash

pip install -r requirements.txt

Run the application:
Bash

    streamlit run app.py

🎯 Use Case

This project serves as a practical demonstration of how Unsupervised Learning can be used as a feature engineering step to enhance Supervised Learning models. It is particularly useful for real estate analytics and spatial data science.
