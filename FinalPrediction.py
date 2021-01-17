import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('sample3.csv')

new_input = pd.read_csv('test.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.01, random_state=4)

print("\nGradient Boosting Accuracy:")
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1,max_features=12, random_state=0).fit(x_train, y_train)

print(100 *gb.score(x_test, y_test))

#gb1=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x_test, y_test)
#print(100 *gb1.score(x_test, y_test))

prediction=gb.predict(x_test) 

print("Confusion Matrix:")
print(confusion_matrix(y_test,prediction))

print("Classification report:")
print(classification_report(y_test,prediction))


print("Gradient Boosting Prediction:")
print(gb.predict((np.array(new_input))))
