import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score#accuracy_score
# import matplotlib.pyplot as plt

df = pd.read_csv(r"House Price Prediction Dataset.csv", encoding="latin1")

df = df[["Bedrooms", "Location", "Price"]]

df = pd.get_dummies(df, columns=["Location"], drop_first=True, dtype=int)
print("Training columns:", df.columns)
print(df.head())

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("mean squared error:", mean_squared_error(y_test,y_pred))
print("R2:",r2_score(y_test,y_pred))


print("\n--- House Price Prediction ---")
room = int(input("Enter number of rooms: "))
location = input("Enter Location: ")

user_data = pd.DataFrame([{"Bedrooms": room, "Location": location}])
user_data_encoded = pd.get_dummies(user_data, columns=["Location"], drop_first=True)
user_data_encoded = user_data_encoded.reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(user_data_encoded)[0]
print(f"Predicted House Price: ₹{predicted_price:,.2f}")

