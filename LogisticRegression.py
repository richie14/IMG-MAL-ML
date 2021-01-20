import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('sample3.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
logistic_regression = LogisticRegression(max_iter=100000)
logistic_regression.fit(x_train, y_train)
#Logistic_Regression(max_iter=10000)

y_pred = logistic_regression.predict(x_test)
accuracy1 = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage1 = 100 * accuracy1
print("Logistic Regression:")
print(accuracy_percentage1)


linearRegression = LinearRegression()
linearRegression.fit(x_train, y_train)
y_pred = linearRegression.predict(x_test)
accuracy2=r2_score(y_test, y_pred)
accuracy_percentage2 = 100 * accuracy2
print("Linear Regression:")
print(accuracy_percentage2)

rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy3=r2_score(y_test, y_pred)
accuracy_percentage3 = 100 * accuracy3
print("RandomForest Regression:")
print(accuracy_percentage3)


new_input = pd.read_csv('test.csv')
#print("Testing Data:")
#print(new_input)

# get prediction for new input
new_output1 = logistic_regression.predict((np.array(new_input)))
# summarize input and output
print("Logistic Regression Prediction:")
print(new_output1)

# get prediction for new input
new_output2 = linearRegression.predict((np.array(new_input)))
# summarize input and output
print("Linear Regression Prediction:")
print(new_output2)

# get prediction for new input
new_output3 = rf.predict((np.array(new_input)))
# summarize input and output
print("Linear Regression Prediction:")
print(new_output3)
