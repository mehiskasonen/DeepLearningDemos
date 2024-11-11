# Self Organizing Map

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for training the SOM
from minisom import MiniSom

# for plotting the SOM.
from pylab import bone, pcolor, colorbar, colorbar, plot, show


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
frauds = np.concatenate((mappings[(3, 8)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)
