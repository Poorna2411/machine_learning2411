from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#genarating data

np.random.seed(24)
n_samples, n_features = 50, 200

X = np.random.randn(n_samples, n_features)

# Generate true coefficients, with only 10 non-zero values
true_coeff = np.zeros(n_features)
true_coeff[:10] = np.random.randn(10)

# Generate target variable (y) with some noise
y = np.dot(X, true_coeff) + np.random.normal(size=n_samples)

X_tarin,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=24)

# Initialize Lasso regression model with a specific regularization strength (alpha)
lasso = Lasso(alpha=0.1)
lasso.fit(X_tarin, y_train)

y_pred = lasso.predict(X_test)


# Calculate Mean Squared Error on test data
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Number of features selected (non-zero coefficients)
n_nonzero_coefs = np.sum(lasso.coef_ != 0)
print(f"Number of selected features: {n_nonzero_coefs}")

# Plot the coefficients
plt.plot(lasso.coef_, marker='o', linestyle='none')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients')
plt.show()
