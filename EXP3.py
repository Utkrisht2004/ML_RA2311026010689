# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
dataset = pd.read_csv(url)

# --- Step 3: Preprocess the data ---
# Linear regression requires all input features to be numeric.
# We need to convert categorical columns ('sex', 'smoker', 'region') into numbers.
# We'll use one-hot encoding, which creates new binary (0 or 1) columns for each category.
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)

# Step 4: Define features (X) and target (y)
# 'y' is the column we want to predict: 'charges'
y = dataset['charges']
# 'X' is all the other columns that we will use for prediction
X = dataset.drop('charges', axis=1)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = regressor.predict(X_test)

# Step 8: Evaluate the model's performance
# R-squared is a metric that tells us how well the model's predictions
# approximate the real data points. A score of 1.0 is a perfect fit.
r2 = r2_score(y_test, y_pred)
print("✅ Model trained successfully!")
print(f"\nModel R-squared score: {r2:.4f}")

# Let's compare a few actual values vs. predicted values
comparison_df = pd.DataFrame({'Actual Charges': y_test, 'Predicted Charges': y_pred})
print("\nSample of Predictions vs Actual Values:")
print(comparison_df.head())

# Step 9: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
# Plot a line representing perfect predictions (y_test = y_pred)
perfect_prediction_line = np.linspace(min(y_test), max(y_test), 100)
plt.plot(perfect_prediction_line, perfect_prediction_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Medical Charges')
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.legend()
plt.grid(True)
plt.show()