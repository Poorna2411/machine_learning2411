import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#step 1:load the data
data = pd.read_csv('advertising_sales.csv')

#step2: perpare the data
amount = data[['Advertising_Spend_(in_$1000)']].values
sales = data['Sales_(in_$1000)'].values

#step3:split the data for tarin and test
X_train, X_test, y_train, y_test = train_test_split(amount, sales, test_size=0.2, random_state=42)

#step4:train the model
model = LinearRegression()
model.fit(X_train,y_train)

#step5 : make predictions

new_amount = np.array([[3.5]])
predict_sales = model.predict(new_amount)
print(f"the sales for the amount $3.5 is : ${predict_sales[0]}")

#step6: testing
y_pre = model.predict(X_test)
mse = mean_squared_error(y_test, y_pre)
print(f"Mean Squared Error on the test set: {mse:.2f}")

#visualize the result

plt.scatter(amount, sales, color='blue')
plt.plot(amount, model.predict(amount), color='red')
plt.show()
