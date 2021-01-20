import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing



df = pd.read_csv('static_analysis.csv')


df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1, random_state=4)

rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)
print("\nRandomForest Regression:")
print(100 * rf.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

prediction=rf.predict((np.array(x_test))) 

print("Confusion Matrix for Static Analysis:")
print(confusion_matrix(y_test,prediction))

tp, fn, fp, tn = confusion_matrix(y_test,prediction,labels=[1,0]).reshape(-1)
print('\nStatisAnalysis')
print('True Positive Rate :', round(tp*100/(tp+fn+fp+tn),2))
print('False Negative Rate :', round(fn*100/(tp+fn+fp+tn),2))
print('False Positive Rate :', round(fp*100/(tp+fn+fp+tn),2))
print('True Negative Rate :', round(tn*100/(tp+fn+fp+tn),2))

print("\nClassification report for Static Analysis:")
print(classification_report(y_test,prediction))


