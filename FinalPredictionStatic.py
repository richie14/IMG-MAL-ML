import pandas as pd
import numpy as np

df = pd.read_csv('static_analysis.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label


from sklearn import metrics 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=4)

from sklearn.ensemble import GradientBoostingClassifier
print("\nGradient Boosting Accuracy for Static Analysis:")
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1,max_features=12, random_state=0).fit(x_train, y_train)

print(100 *gb.score(x_test, y_test))

#gb1=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x_test, y_test)
#print(100 *gb1.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

prediction=gb.predict(x_test) 

print("Confusion Matrix for Static Analysis:")
print(confusion_matrix(y_test,prediction))

tp, fn, fp, tn = confusion_matrix(y_test,prediction,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', tp*100/(tp+fn+fp+tn))
print('False Negative Rate :', fn*100/(tp+fn+fp+tn))
print('False Positive Rate :', fp*100/(tp+fn+fp+tn))
print('True Negative Rate :', tn*100/(tp+fn+fp+tn))

print("\nClassification report for Static Analysis:")
print(classification_report(y_test,prediction))

new_input = pd.read_csv('test.csv')
print("Gradient Boosting Prediction:")
print(gb.predict((np.array(new_input))))





