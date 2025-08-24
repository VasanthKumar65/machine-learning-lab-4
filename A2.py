"""A2. Calculate MSE, RMSE, MAPE and R2 scores for the price prediction exercise done in Lab 02. 
Analyse the results. """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\Vasanth Kumar\Downloads\Housing.csv")

# Choose target column (change if different in your file)
target = "price"   # <-- update this if your dataset has a different price column
X = df.drop(columns=[target])
y = df[target]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAPE = {mape:.2f}%")
print(f"R²   = {r2:.4f}")
