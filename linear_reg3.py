import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('salesData.csv')

#  in the dataset adversing and price are independant and sales are dependent
X = data[['Advertising', 'Price']].values
y = data['Sales'].values

X_train, X_test, y_train,y_test = train_test_split(X, y, train_size=0.2, random_state=24 )

model = LinearRegression()
model.fit(X_train,y_train)

y_pre = model.predict(X_test)
mse = mean_squared_error(y_test,y_pre)
print("mse is :", mse)

predict = model.predict([[15,5]])
print("pridiction for 15,5 is:",predict[0])
