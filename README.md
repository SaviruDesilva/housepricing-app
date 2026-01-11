🏠 House Price Prediction with Clustering

An end-to-end Machine Learning web application that predicts house prices by combining Unsupervised Learning (K-Means Clustering) and Supervised Learning (Random Forest Regression).

By grouping houses based on geographic coordinates (Latitude/Longitude), the model captures location-based trends more effectively, leading to significantly more accurate price estimations.

🔗 View Live Demo
✨ Features

    Hybrid ML Approach: Integrates K-Means labels into a Random Forest Regressor to capture spatial patterns.

    Interactive UI: A clean, responsive interface built with Streamlit for real-time predictions.

    Data-Driven Insights: Trained on the classic California Housing Dataset.

    Model Persistence: Features, scalers, and models are serialized using joblib for instant inference.

    Automated Pipeline: Robust preprocessing including median imputation and StandardScaler integration.

🧠 Machine Learning Workflow

The project follows a modular pipeline to ensure data integrity and model performance:

    Data Preprocessing: Cleaning the California Housing dataset and handling missing values via median imputation.

    Feature Engineering: * K-Means Clustering: Generating geographic clusters based on longitude and latitude to act as a "neighborhood" feature.

    Scaling: Standardizing numerical features using StandardScaler.

    Model Training: Training a Random Forest Regressor on the augmented feature set.

    Evaluation: Validating performance using the R2 Score and RMSE.

    Deployment: Exporting the pipeline as .pkl files and serving via Streamlit.

🛠️ Tech Stack
Category	Tools
Frontend	Streamlit
Language	Python 3.x
ML Framework	Scikit-learn
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Serialization	Joblib
