import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import classification_report

dataset = np.loadtxt('test_last.csv', delimiter=";")

X = dataset[:,0]
y = dataset[:,1]

X = X.reshape(-1,1)
#normalized_X = preprocessing.normalize(X)
#standardized_X = preprocessing.scale(X)

train_data, test_data, train_labels, test_labels = model_selection.train_test_split(X, 
                                                                                    y, 
                                                                                    test_size = 0.3,
                                                                                    random_state = 1)

model = svm.SVC(verbose=True)
model.fit(train_data,train_labels)

result = model.predict(test_data)
for j in range(len(test_data)):
    print("Data: %s, Even: %s" % (test_data[j], result[j]))

print(model.score(test_data,test_labels))
print(test_labels)
print(result)
print(classification_report(test_labels, result))
