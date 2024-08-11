import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#step1: load the dataset
data = pd.read_csv('house_prices.csv')

#step2: prepare the data
h_size = data[['Size (sq ft)']].values
h_prices = data['Price ($)'].values


#step3: Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(h_size, h_prices, test_size = 0.2, random_state=42)


#step4:Train the model
model = LinearRegression()
model.fit(X_train, y_train)


#step5: make the predictions
new_h_size = np.array([[1500]])
predicted_price = model.predict(new_h_size)
print(f"Predicated price for a 2500 sq ft house : ${predicted_price}")
