# import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import numpy as np


# read the dataset
df = pd.read_csv('sample3.csv')
df.head()


print(df.shape)

num_of_classes = len(df.Label.unique())
#print(num_of_classes)

#df.describe()


# split train input and output data
#X = df.drop(axis=0, columns=['Label', 'Case #'])
X = df.drop(axis=0, columns=['Label'])
Y = df.Label

#Print the shape of X and Y
#print(X.shape)
#print(Y.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# Create a classifier
xgb = XGBClassifier(booster='gbtree', objective='multi:softprob', use_label_encoder=False, random_state=42, eval_metric="auc", num_class=num_of_classes)

# Fit the classifier with the training data
xgb.fit(X_train,y_train)


# Use trained model to predict output of test dataset
val = xgb.predict(X_test)

lb = preprocessing.LabelBinarizer()
lb.fit(y_test)

y_test_lb = lb.transform(y_test)
val_lb = lb.transform(val)

print("MultiClass Classifier:")
print(roc_auc_score(y_test_lb, val_lb, average='macro'))


output = pd.DataFrame()
output['Expected Output'] = y_test
output['Predicted Output'] = val
output.head()






new_input = pd.read_csv('test.csv')
#print("Testing Data:")
#print(new_input)

# get prediction for new input
new_output = xgb.predict((np.array(new_input)))

# summarize input and output
print("Final Prediction:")
print(new_output)


