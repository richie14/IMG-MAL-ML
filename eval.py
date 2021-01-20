# example of creating a test dataset
from sklearn.datasets import make_blobs
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# summarize the shape of the arrays
print(X.shape, y.shape)


# fit a logistic regression on the training dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
# create the inputs and outputs
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# evaluate predictions
acc = accuracy_score(y, yhat)
print(acc)



# define input
new_input = [[2.12309797, -1.41131072]]

# get prediction for new input
new_output = model.predict(new_input)

# summarize input and output
print(new_input, new_output)



# make predictions on the entire training dataset
yhat = model.predict(X)

# connect predictions with outputs
for i in range(10):
	print(X[i], yhat[i])
