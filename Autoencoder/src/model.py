import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


"""Imports the dataset."""
movies = pd.read_csv('Dataset/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('Dataset/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('Dataset/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')


"""
Prepare the training set and the test set:
- The training set and test set are used to train and evaluate the model, respectively.
- The data is read and converted into a numpy array with integer type.
"""
training_set = pd.read_csv('Dataset/ml-100k/u1.base', delimiter='\t', header=None)
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('Dataset/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

"""
Takes the number of users and movies.
"""
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


"""
Converts the data into an array with users in lines and movies in columns.
The cell values are the ratings given by the user to the movie. If a user has not rated a movie, the cell value is zero.
"""


def convert(data):
    new_data = []
    for user in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == user]
        id_ratings = data[:, 2][data[:, 0] == user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)


"""Converts the data into Torch tensors."""
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""Creates the architecture of the neural network."""


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # first full connection
        self.fc2 = nn.Linear(20, 10) # second layer
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    """Forward propagation. Proceeds the different encodings and decodings when the observation
     is forwarded into the network."""
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)  # lr - learning rate

"""Trains the SAE neural network.
- The model is trained for a specified number of epochs.
- During training, the model learns to reconstruct the input data (user ratings).
- The loss is calculated using only non-zero ratings, and the model parameters are updated accordingly.
"""
nr_epochs = 200
for epoch in range(1, nr_epochs + 1):
    train_loss = 0
    s = 0.  # used to compute RMSE (Root mean square error)
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.requires_grad = False  # Gradiant is computed with respect to input and not target. Optimizes memory.
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
    print('Epoch: ' + str(epoch) + ', Loss: ' + str(train_loss / s))

"""Tests the SAE neural network.
- The trained model is evaluated on the test set.
- The loss is calculated similarly to the training phase.
"""
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0) # Ensure target has the same shape as input
    if torch.sum(target.data > 0) > 0:
        output = sae(input)  # vector of predicted ratings the user hasn't watched yet.
        target.requires_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print('Test loss: ' + str(test_loss / s))
