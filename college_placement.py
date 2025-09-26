# Importing necessary libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv("CollegePlacement.csv")
print(df.head())
print(df.info())

# Checking for null values
print(df.isnull().sum())

# Dropping unnecessary columns - College_ID
df = df.drop(columns=["College_ID"])

# Encoding the categorical variables - Label Encoding
label_enc = LabelEncoder()
df["Internship_Experience"] = label_enc.fit_transform(df["Internship_Experience"])
df["Placement"] = label_enc.fit_transform(df["Placement"])

# Defining features and target variable
X = df.drop(columns=["Placement", "CGPA"])   # Drop CGPA instead of Prev_sem_results
y = df["Placement"]

# Spilitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Creation and Training - Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Accuracy Check
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy :: ", accuracy)
print(report)