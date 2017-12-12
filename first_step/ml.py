import numpy as np
import pickle
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

dataset = np.loadtxt('learn_1.csv', delimiter=";", skiprows=1)

X = dataset[:,0]
y = dataset[:,1]

X = X.reshape(-1,1)
#normalized_X = preprocessing.normalize(X)
#standardized_X = preprocessing.scale(X)

#model = ExtraTreesClassifier()
#model.fit(X, y)
## display the relative importance of each attribute
#print(model.feature_importances_)

model = LogisticRegression()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

test_dataset = np.loadtxt('study.csv', delimiter=";", skiprows=1)
#X = test_dataset[:,0]
test_X = test_dataset
print(test_X)
#y = test_dataset[:,1]
test_X = test_X.reshape(-1,1)

result = loaded_model.score(X, y)
print(result)

predictions = loaded_model.predict(test_X)
for j in range(len(test_X)):
    print("Data: %s, Even: %s" % (test_X[j], predictions[j]))
