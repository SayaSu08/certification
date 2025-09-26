import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = pd.read_csv("data.csv")
print(df.head())
print(df.info())

# Drop unnecessary columns
df = df.drop(columns=["floors", "waterfront", "view", "condition", "statezip"])
df = df.dropna()

# Define features and target
X = df[["sqft_living", "bedrooms", "bathrooms", "sqft_lot",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:10])  # show first 10 predictions

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)