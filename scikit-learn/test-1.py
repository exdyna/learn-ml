# this example is from https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# An introduction to machine learning with scikit-learn¶
# 

# 1. training set and testing set

# Loading an example dataset

from sklearn import datasets
from sklearn import svm
import pickle

iris = datasets.load_iris()
digits = datasets.load_digits()

# print(digits.data)
# print(iris.data)

# print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])

clf.predict(digits.data[-1:])
print(clf)

# Model persistence

X, y = iris.data, iris.target
clf.fit(X,y)
print("clf", clf)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

print("clf2",clf2)
print(y[0])

# it may be more interesting to use joblib’s replacement for pickle 
# (joblib.dump & joblib.load), which is more efficient on big data 
# but it can only pickle to the disk and not to a string
import joblib
joblib.dump(clf, 'filename.joblib') 
clf3 = joblib.load('filename.joblib')
clf3.predict(X[0:1])
print("clf3",clf3)

