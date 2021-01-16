import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing


df = pd.read_csv('sample3.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=4)


loR = LogisticRegression(max_iter=100000)
loR.fit(x_train, y_train)
print("\nLogistic Regression:")
print(100 * loR.score(x_test, y_test))

liR = LinearRegression()
liR.fit(x_train, y_train)
print("\nLinear Regression:")
print(100 * liR.score(x_test, y_test))

rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)
print("\nRandomForest Regression:")
print(100 * rf.score(x_test, y_test))

decisionTree = tree.DecisionTreeClassifier(max_depth=5)
decisionTree.fit(x_train, y_train)
y_pred = decisionTree.predict(x_test)
accuracy4 = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage4 = 100 * accuracy4
print("\nDecision tree:")
print(accuracy_percentage4)
print(100 *decisionTree.score(x_test, y_test))


print("\nGradient Boosting:")
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x_train, y_train)
print(100 *gb.score(x_test, y_test))


nb = GaussianNB()
nb.fit(x_train, y_train)
print("\nGradient Boosting:")
print(100 *nb.score(x_test, y_test))


knn =  KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print("\nK-Nearest Neighbour:")
print(100 *knn.score(x_test, y_test))


sv =  SVC(probability=True)
sv.fit(x_train, y_train)
print("\nSVM:")
print(100 *sv.score(x_test, y_test))

num_of_classes = len(df.Label.unique())
xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', use_label_encoder=False, random_state=42, eval_metric="auc", num_class=num_of_classes)
xgb.fit(x_train,y_train)
val = xgb.predict(x_test)
lb = preprocessing.LabelBinarizer()
lb.fit(y_test)
y_test_lb = lb.transform(y_test)
val_lb = lb.transform(val)

print("\nMultiClass Classifier:")
print(roc_auc_score(y_test_lb, val_lb, average='macro'))

