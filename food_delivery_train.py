import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from math import radians, sin, cos, sqrt, atan2
import joblib
import os


# Haversine Distance Function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# Load Data
df = pd.read_csv("train.csv")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# Data Cleaning
df = df.dropna()
df = df[df["Delivery_person_Age"] != 'NaN ']
df["Delivery_person_Age"] = df["Delivery_person_Age"].astype(int)
df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].astype(float)
df["multiple_deliveries"] = df["multiple_deliveries"].replace('NaN ', 0).astype(int)
df["Time_taken(min)"] = df["Time_taken(min)"].str.extract("(\d+)").astype(int)


# Feature Engineering
# Compute real distance
df["distance_km"] = haversine(df["Restaurant_latitude"], df["Restaurant_longitude"],
                              df["Delivery_location_latitude"], df["Delivery_location_longitude"])

# Drop unneeded IDs
df = df.drop(columns=["ID", "Delivery_person_ID", "Order_Date", "Time_Orderd", "Time_Order_picked"])

# Encode categoricals
cat_cols = df.select_dtypes(include='object').columns
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c])

# KMeans clustering (segment city areas)
coords = df[["Restaurant_latitude", "Restaurant_longitude"]]
kmeans = KMeans(n_clusters=5, random_state=42)
df["region_cluster"] = kmeans.fit_predict(coords)

# Save cluster model
os.makedirs("models", exist_ok=True)
joblib.dump(kmeans, "models/kmeans_region.pkl")

# Split Features & Target
X = df.drop(columns=["Time_taken(min)"])
y = df["Time_taken(min)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")

# Model Training + Grid Search
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
print("\nBest RF Parameters:", grid.best_params_)

# XGBoost as alternate
xgb.fit(X_train_scaled, y_train)

# Evaluation
def evaluate(model, name):
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n{name} → RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
    return preds

rf_preds = evaluate(best_rf, "Random Forest (Tuned)")
xgb_preds = evaluate(xgb, "XGBoost")

# Save best model
joblib.dump(best_rf, "models/best_model.pkl")

# Visualization
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=rf_preds, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Random Forest")
plt.savefig("models/actual_vs_pred.png")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(y_test - rf_preds, bins=30, kde=True, color='skyblue')
plt.title("Error Distribution (RF)")
plt.savefig("models/error_distribution.png")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=best_rf.feature_importances_, y=X.columns)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()

