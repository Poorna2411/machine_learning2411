import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
n_samples = 100
X = 2.5 * np.random.randn(n_samples) + 6  # Number of rooms (around 6 rooms on average)
y = 50 * X + np.random.randn(n_samples) * 20  # House price (in thousands) with some noise

# Reshape X to make it a 2D array, as required by scikit-learn
X = X.reshape(-1, 1)

# Create a DataFrame for better visualization (optional)
data = pd.DataFrame({
    'Rooms': X.flatten(),
    'Price': y
})

data.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2:.2f}')

# Plot the training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')
# Plot the test data
plt.scatter(X_test, y_test, color='green', label='Test Data')
# Plot the regression line
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Number of Rooms')
plt.ylabel('House Price (in thousands)')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
