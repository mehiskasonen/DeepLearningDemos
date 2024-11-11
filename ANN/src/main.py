import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


"""
Part 1. Preprocessing the dataset
"""


"""
importing the dataset
"""
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# label encoding the "Gender" column. Will give Female the value of 0, and male the value of 1.
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

# One Hot Encoding of geography column.
# 1.0 0.0 0.0 will correspond to France
# 0.0 0.0 1.0 to Spain etc.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(x))

# Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling for both Training Set and Test Set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
Part 2 Building the ANN
"""


"""
Initializing the ANN
as a sequence of layers.
"""
ann = tf.keras.models.Sequential()


"""
Adding the input layer and the first hidden layer
relu - stands for rectifier activation function.
"""
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""
Adding the second hidden layer
"""
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


"""
Adding the output layer
uses a sigmoid activation function
When doing non-binary classification, we can use a softmax activation function.
"""
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



"""
Part 3. Training the ANN
"""


"""
Compiling the ANN
Adam optimizer performs stochastic gradient descent, which will update the weights of the neural network in order to reduce
loss error between predictions and results. When training the ANN on the training set, we will at each iteration compare the
predictions in a batch to the real results in the same batch. Optimizer will update the weights to reduce the loss on the next
iteration.

Loss function - way to compute the difference between the predictions and the true values. For binary classification, it must be
binary_crossentropy. For non-binary classification, it must be cross_entropy_loss
"""
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


"""
Training the ANN on the Training Set
batch - the number of predictions to be in the batch to compare with the true values of the training set.
Default value is 32.
"""
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

"""
Part 4. Making the predictions and valuating the model.
"""

"""
Predicting the results of a single observation
"""
if __name__ == '__main__':
    prediction = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
    # The model will output a probability, so we round it to get the binary outcome
    result = (prediction > 0.5).astype(int)
    print(f'The customer will {"leave" if result[0][0] == 1 else "not leave"} the bank.')
    #print(x)
    print("--")


"""
Predicting the Test set results.
"""
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

"""
Making the confusion matrix.
"""
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(accuracy)

