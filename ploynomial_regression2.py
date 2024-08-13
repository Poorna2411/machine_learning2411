import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data 
hours_studied = np.array([1,2,3,4,5]).reshape(-1,1)
test_scores = np.array([2,8,18,32,50])

#polynomial features
poly = PolynomialFeatures(degree=2)
hours_poly = poly.fit_transform(hours_studied)

#fit the model
model = LinearRegression()
model.fit(hours_poly, test_scores)

# Predictions
hours_range = np.linspace(1,5,100).reshape(-1,1)
hours_range_ploy = poly.transform(hours_range)
predictions = model.predict(hours_range_ploy)

# Plot
plt.scatter(hours_studied, test_scores, color='red', label='Data Points')
plt.plot(hours_range, predictions, color='blue', label='Polynomial Fit')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Polynomial Regression')
plt.show()