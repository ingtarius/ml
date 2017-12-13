import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# Load data
dataset = pd.read_csv('learn.csv', delimiter=';')

print(dataset.info())

# set targets and features

target = ['EVEN']
features = ['NUMBER', 'FEATURE']

categorical_features = dataset.columns.difference(target + features)

x = dataset[features]
y = dataset[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print(model)

# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# Real data
dataset_pred = pd.read_csv('test.csv', delimiter=';')
x_pred = dataset_pred[features]
predicted = model.predict(x_pred)

submission = pd.DataFrame({
    "NUMBER": dataset_pred["NUMBER"],
    "EVEN": predicted
})

print(submission)
