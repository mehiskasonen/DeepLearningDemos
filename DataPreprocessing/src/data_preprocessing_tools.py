"""
Simple example of how to import dataset of a retail company that including clients country of origin,
age, salary and outcome if product was purchased or not. Also demonstrates how to split data into
training set and test set and finally how to apply feature scaling - a process of normalizing the values
of the features in the dataset so that they fall within a similar range. This is particularly important
when the dataset contains features with varying units or magnitudes, as it helps ensure that each
feature contributes equally to the model's learning process.
"""

import numpy as np
import pandas as pd

"""Imports the dataset and extracts variables."""
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Extracts the independent variables (all columns except the last) into matrix X.
y = dataset.iloc[:, -1].values  # Extracts the dependent variable (the last column) into vector y.
print(X)
print(y)

"""Handles missing data in the dataset."""
from sklearn.impute import SimpleImputer

# Creates an imputer object that replaces missing values with the mean of the column
# This imputer uses the mean of the non-missing values in each column to replace any missing values,
# ensuring that the dataset remains complete and usable for further processing or modeling.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fitting the imputer to the columns with missing values (columns 1 and 2 in this case)
imputer.fit(X[:, 1:3])

# Applying the transformation to replace missing values in these columns
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

"""
Encoding categorical data for the independent variables.
Using OneHotEncoder to convert the 'Country' column (index 0) into a one-hot encoded format.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Creating a column transformer with OneHotEncoder for the 'Country' column and passthrough for others.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Applying the column transformer and converting the result to a numpy array.
X = np.array(ct.fit_transform(X))
print(X)
# Encoding the Dependent Variable 'Purchased' as numeric values.
from sklearn.preprocessing import LabelEncoder

# Creates a label encoder object
le = LabelEncoder()

# Fits and transforms the dependent variable to numeric values (Yes/No to 1/0).
y = le.fit_transform(y)
print(y)

"""Splitting the dataset into the Training set and Test set."""
from sklearn.model_selection import train_test_split

# Splitting the data: 80% for training and 20% for testing, with a random seed of 1 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

"""Applies feature scaling to the dataset"""
from sklearn.preprocessing import StandardScaler

# Creating a standard scaler object.
sc = StandardScaler()

# Fitting the scaler to the training data (excluding the first three columns which
# include the one-hot encoded country data)
# and transforming the data to have a mean of 0 and a standard deviation of 1.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

# Transforming the test set using the same scaler fitted to the training set.
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)
