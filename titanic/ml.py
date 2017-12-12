import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing


# Load data
dataset = pd.read_csv('train.csv', usecols = [1,2,4,5,6,7])

# Replace binary data Male/Female
dataset = dataset.replace({'Sex': {'male': '1', 'female': '0'}})

# Skip NaN rows
dataset = dataset.dropna()

# Set target and features
target = ['Survived']
numerical_features = ['Pclass', 'Sex', 'Age', 'SibSp']
categorical_features = dataset.columns.difference(target + numerical_features)


x = dataset[numerical_features]
y = dataset[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)

print(classification_report(y_train, y_pred))

# make predictions
print(clf.score(x_test, y_test))
