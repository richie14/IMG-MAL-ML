import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

data = pd.read_csv('sample2.csv')
#train.info()
y = data['Label']
X = data.drop(['Label'], axis = 1)


X.isnull().sum()

X.shape


#train.drop('Label',axis=1,inplace=True)
#train.dropna(inplace=True)


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train.drop('Label',axis=1),train['Label'], test_size=0.30,random_state=101)


logmodel = LogisticRegression()
#logmodel.fit(X_train,y_train)
#predictions = logmodel.predict(X_test)

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(logmodel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)

best_clf.best_estimator_
print (f'Accuracy - : {best_clf.score(X,y):.3f}')
