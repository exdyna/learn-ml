# this example is from https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# An introduction to machine learning with scikit-learn¶
# 

# 1. training set and testing set

# Loading an example dataset

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
print(iris.data)
