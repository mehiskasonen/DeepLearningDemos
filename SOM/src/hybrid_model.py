# Hybrid Deep Learning Model combining ANN and SOM

# Part 1 - identifying credit data frauds with SOM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, colorbar, plot, show
import tensorflow as tf


# Statlog (Australian Credit Approval) Data set
dataset = pd.read_csv("DataSet/Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# Step 3. Randomly initialize the values of the weight vectors to small numbers close to 0 (but not 0)
som.random_weights_init(X)

# Step 4 - 9
som.train_random(X, 100)

# Visualize the result to identify the outlying neurons inside the map. Plot the SOM.
bone()  # initialize the plot
pcolor(som.distance_map().T)  # Distance map as background + transpose the matrix
colorbar()  # color legend

# plot markers for the clusters
markers = ['o', 's']
colors = ['r', 'g']

# red circle if customer didn't get approval. Green square if customer got approval.
for i, x in enumerate(X):
    w = som.winner(x)  # Get the winning node
    plot(w[0] + 0.5, w[1] + 0.5, markers[Y[i]], markeredgecolor=colors[Y[i]],
         markerfacecolor='None', markersize=10, markeredgewidth=2)

show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5, 3)], mappings[(5, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependant variables
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1


# Feature scaling for both Training Set and Test Set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


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
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))


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
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


"""
Training the ANN on the Training Set
batch - the number of predictions to be in the batch to compare with the true values of the training set.
Default value is 32.
"""
ann.fit(customers, is_fraud, batch_size=1, epochs=2)


"""
Predicting the probabilities of frauds.
"""
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]


if __name__ == '__main__':
    print(y_pred)
