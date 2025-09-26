import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("data.csv")
print(df.head())
print(df.info())

# Drop unnecessary columns
df = df.drop(columns=["floors", "waterfront", "view", "condition", "statezip"])
df = df.dropna()

# Features & Target
X = df[["sqft_living", "bedrooms", "bathrooms", "sqft_lot", 
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]]
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Creation and Training - Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Custom Accuracy (within ±10% tolerance)
tolerance = 0.10
custom_accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < tolerance)

print("\nModel Performance:")
print("MAE       :: ", mae)
print("RMSE      :: ", rmse)
print("R² Score  :: ", r2)
print("Custom Accuracy (within 10%) :: ", custom_accuracy)

# --- Feature Importance Plot ---
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting House Price")
plt.show()

# --- Actual vs Predicted Scatter Plot ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
