import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.4,random_state=2411)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pre = model.predict(X_test)

conf_matr = confusion_matrix(y_test,y_pre)

print(conf_matr)
