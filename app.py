import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


st.set_page_config(page_title="🍔 Food Delivery Time Predictor", layout="centered")

st.title("🍽️ Food Delivery Time Prediction App")

# Load models
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")
kmeans = joblib.load("models/kmeans_region.pkl")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

st.subheader("Enter Order Details:")

Delivery_person_Age = st.number_input("Delivery Person Age", 18, 65, 30)
Delivery_person_Ratings = st.slider("Delivery Person Rating", 0.0, 5.0, 4.5)
Restaurant_latitude = st.number_input("Restaurant Latitude", 8.0, 35.0, 12.97)
Restaurant_longitude = st.number_input("Restaurant Longitude", 70.0, 90.0, 77.59)
Delivery_location_latitude = st.number_input("Delivery Location Latitude", 8.0, 35.0, 12.99)
Delivery_location_longitude = st.number_input("Delivery Location Longitude", 70.0, 90.0, 77.60)
Weatherconditions = st.selectbox("Weather", ["Sunny", "Stormy", "Cloudy", "Windy", "Fog"])
Road_traffic_density = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
Vehicle_condition = st.slider("Vehicle Condition", 0, 2, 1)
Type_of_order = st.selectbox("Order Type", ["Meal", "Snack", "Drinks"])
Type_of_vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])
multiple_deliveries = st.selectbox("Multiple Deliveries?", [0,1])
Festival = st.selectbox("Festival?", ["Yes", "No"])
City = st.selectbox("City", ["Urban", "Metropolitan", "Semi-Urban"])

if st.button("🚴 Predict Delivery Time"):
    distance = haversine(Restaurant_latitude, Restaurant_longitude, Delivery_location_latitude, Delivery_location_longitude)
    region_cluster = kmeans.predict([[Restaurant_latitude, Restaurant_longitude]])[0]
    
    # Prepare input
    data = pd.DataFrame([[
        Delivery_person_Age, Delivery_person_Ratings,
        Restaurant_latitude, Restaurant_longitude,
        Delivery_location_latitude, Delivery_location_longitude,
        Weatherconditions, Road_traffic_density, Vehicle_condition,
        Type_of_order, Type_of_vehicle, multiple_deliveries,
        Festival, City, distance, region_cluster
    ]], columns=[
        'Delivery_person_Age','Delivery_person_Ratings','Restaurant_latitude','Restaurant_longitude',
        'Delivery_location_latitude','Delivery_location_longitude','Weatherconditions','Road_traffic_density',
        'Vehicle_condition','Type_of_order','Type_of_vehicle','multiple_deliveries','Festival','City','distance_km','region_cluster'
    ])
    
    # Encode categoricals same as before
    data_encoded = pd.get_dummies(data).reindex(columns=columns, fill_value=0)
    data_scaled = scaler.transform(data_encoded)
    
    pred = model.predict(data_scaled)[0]
    st.success(f"⏱️ Estimated Delivery Time: **{pred:.2f} minutes**")

