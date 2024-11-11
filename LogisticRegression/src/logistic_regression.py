"""Example of logistic regression on car retailer data (age, estimated salary, purchase decision)
which customer is most likely to buy a new car. Used for targeting future customers."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Imports the dataset and extracts variables."""
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # Extracts the independent variables (all columns except the last) into matrix X.
y = dataset.iloc[:, -1].values  # Extracts the dependent variable (the last column) into vector y.
# print(X)
# print(y)

"""Splitting the dataset into the Training set and Test set."""
from sklearn.model_selection import train_test_split

# Splitting the data: 80% for training and 20% for testing, with a random seed of 1 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


"""Applies feature scaling to the dataset"""
from sklearn.preprocessing import StandardScaler

# Creating a standard scaler object.
sc = StandardScaler()

# Fitting the scaler to the training data (excluding the first three columns which
# include the one-hot encoded country data)
# and transforming the data to have a mean of 0 and a standard deviation of 1.
X_train = sc.fit_transform(X_train)

# Transforming the test set using the same scaler fitted to the training set.
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)

"""
Training the Logistic Regression Model on Training set.
Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# print(y_pred)
# print(y_test)

"""
Predicting a new result.
Age: 30
Predicted salary: 87,000
Yes - if prediction == 1
No - if prediction == 0
"""
x_pred = classifier.predict(sc.transform([[30, 87000]]))
if x_pred == 1:
    print('Yes')
else:
    print('No')

"""
Predicting the Test set results.
Displays vector of decision boundaries.
"""
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


"""Makes the confusion matrix.
Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
Shows the number of true positives, true negatives, false positives, and false negatives in a tabular format.
Returns: [
[True Negative False Positive]
[False Negative True Positive]
]
Accuracy of predictions: %
"""
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

"""
Visualizing the Training set results.
Green points - customers who bought a car.
Red points - customers who didn't buy a car.
Green on red - false negatives
Red on green - false positives
"""
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""Visualizing the Test set results."""
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()