import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#getting data

data = pd.read_csv('employee_salaries.csv')
exp = data[['YearsExperience']].values
salary = data['Salary'].values
#spliting data for model

X_train,X_test,y_train,y_test = train_test_split(exp, salary, train_size=0.2, random_state=42)

#defining model

model = LinearRegression()
model.fit(X_train,y_train)

#prediting
newE = np.array([[11]])
y_pre = model.predict(newE)
y_pre.reshape(-1,1)
print(f'salary for 11 year employ is:{y_pre}')

