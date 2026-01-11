# 🏠 House Price Prediction with Clustering

A comprehensive Machine Learning web application built with **Streamlit** that predicts house prices. By integrating geographical clustering with regression analysis, this tool provides more localized and accurate real estate valuations.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housepricing-app-g6qvpmqmnbvcpsepzf6oh3.streamlit.app/)

## 📖 Project Overview

Real estate value is heavily dependent on location. This application uses a **hybrid approach**: it first groups properties into geographical neighborhoods using **KMeans Clustering** (based on Latitude and Longitude) and then applies a **Random Forest Regressor** to predict the final price. This method allows the model to "understand" specific regional price trends.

### Key Features

* 📊 **Interactive Data Exploration:** View dataset statistics and distributions (e.g., Income vs. House Price).
* 🌍 **Location-Based Clustering:** Automatically identifies geographical clusters to improve prediction accuracy using KMeans.
* 🤖 **AI Prediction:** Real-time prediction of house prices using a Random Forest Regressor trained on regional data.
* 💾 **Efficient Model Persistence:** Uses Joblib to save and load models for fast, real-time inference without retraining.
* 📈 **Model Performance:** Visualizes the model's accuracy and evaluation metrics like the R² score.

## 💻 Tech Stack

* **Python 3.x**
* **Streamlit** (Web Interface)
* **Scikit-Learn** (Machine Learning - Random Forest & KMeans)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Joblib** (Model Serialization)

## 🚀 How to Run Locally

If you want to run this app on your own computer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/house-pricing-app.git](https://github.com/YOUR_USERNAME/house-pricing-app.git)
    cd house-pricing-app
    ```

2.  **Install dependencies:**
    Make sure you have a `requirements.txt` file, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

```text
├── housing.csv          # The dataset containing housing features
├── app.py               # The main Streamlit application code
├── requirements.txt     # List of Python libraries required
└── README.md            # Project documentation
