# Food-Delivery-Time-Prediction
This project builds a machine learning model to predict the delivery time (in minutes) for food orders based on real-world delivery data. It includes distance computation using the Haversine formula, K-Means clustering for regional segmentation, hyperparameter tuning, and a Streamlit web app for interactive prediction.


# Project Overview

Key Objectives

  1. Predict delivery time using location, traffic, weather, and order features.
  2. Compute real geographic distance using the Haversine formula.
  3. Segment delivery regions using K-Means clustering.
  4. Apply GridSearchCV for model tuning.
  5. Deploy prediction model using Streamlit.

# Models Used

1. Random Forest Regressor (tuned) – primary predictive model

2. XGBoost Regressor – comparison model

# Setup Instructions

Install Python

Use Python 3.11.x (recommended).

pip install pandas numpy scikit-learn xgboost seaborn matplotlib streamlit joblib

# Model Training

Run the training pipeline to clean data, compute distance, cluster delivery regions, tune models, and save outputs.

python food_delivery_train.py

# What It Does

  1. Loads and cleans the dataset (train.csv)
  2. Encodes categorical features
  3. Calculates real delivery distance using Haversine
  4. Clusters restaurants into 5 regions using KMeans
  5. Splits data into training/testing sets
  6. Scales numeric features
  7. Performs GridSearchCV on Random Forest
  8. Trains XGBoost as an alternative
  9. Evaluates models with RMSE, MAE, and R²
  10. Saves models and plots to /models/

Loaded: 45593 rows, 20 columns
Best RF Parameters: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
Random Forest (Tuned) → RMSE=3.70, MAE=2.99, R2=0.842
XGBoost → RMSE=3.82, MAE=3.05, R2=0.831
✅ Training complete. Models & visualizations saved in /models/

# Visualization Outputs

1. After training, the following plots are saved in the /models/ folder:

2. actual_vs_pred.png → Actual vs. Predicted times

3. error_distribution.png → Residual error histogram

4. feature_importance.png → Top predictive features

# Streamlit App

Launch the interactive app to predict delivery times based on user inputs.

Run the app

streamlit run app.py

# What it does

- Takes delivery and order details as input

- Computes real distance (km) and regional cluster automatically

- Predicts estimated delivery time using the trained model

- Displays results in an intuitive, real-time interface

3 Example Interface

Input: Rider age, ratings, location coordinates, traffic level, vehicle type, etc.

Output:

⏱️ Estimated Delivery Time: 23.4 minutes

Key Features Recap

| Feature                   | Description                                   |
| ------------------------- | --------------------------------------------- |
| **Distance Calculation**  | Haversine formula for accurate Earth distance |
| **Clustering**            | K-Means for geographic segmentation           |
| **Hyperparameter Tuning** | GridSearchCV for Random Forest optimization   |
| **Comparison Model**      | XGBoost for performance benchmarking          |
| **Scalability**           | Works on 45K+ records                         |
| **Deployment**            | Streamlit app for end-user prediction         |

# Technologies Used

Python 3.11

Pandas / NumPy — Data wrangling

Scikit-learn — Modeling, preprocessing, tuning

XGBoost — Gradient boosting model

Seaborn / Matplotlib — Visualization

Joblib — Model persistence

Streamlit — Web app deployment

# Results Summary

| Model                 | RMSE | MAE  | R²        |
| --------------------- | ---- | ---- | --------- |
| Random Forest (Tuned) | 3.70 | 2.99 | **0.842** |
| XGBoost               | 3.82 | 3.05 | 0.831     |

# Future Enhancements

Add deep learning (LSTM) for time-based prediction

Integrate real-time traffic/weather APIs

Deploy model to the cloud (e.g., Streamlit Cloud, Render, or AWS)

# Author

Ittyavira C Abraham

MCA (AI) — Amrita Vishwa Vidyapeetham
