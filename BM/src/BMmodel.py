"""
Boltzman machine example.
Predicts if user will like a movie or not.
"""

import numpy as np
import pandas as pd
import torch.nn.parallel
import torch.utils.data

"""
Imports the dataset
"""
movies = pd.read_csv('Dataset/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('Dataset/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('Dataset/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

"""
Prepares the training set and test set.
"""
training_set = pd.read_csv('Dataset/ml-100k/u1.base', delimiter='\t')
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

"""Converts the ratings to binary ratings 1 (Liked), 0 (Not Liked)."""
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

"""Creates the architecture of the neural network."""


class RBM:

    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    """Return all the probabilities and samplings of the hidden neurons, given the values of the visible nodes."""

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    """Return all the probabilities and samplings of the visible neurons, given the values of the hidden nodes."""

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    """
    v0 - input vector containing all the ratings of all the movies by one user (input vector of observation)
    vk - visible nodes obtained after k-samplings (after round-trips eg k-iterations and k-contrastive-divergence)
    ph0 - vector of probabilities that at the first iterations the hidden nodes equal 1 given the values of v0
    phk - probabilities of the hidden nodes after k-sampling given the values of the hidden nodes vk.
    """

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

"""Trains the RBM."""

nb_epochs = 10
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: ' + str(epoch) + ', loss: ' + str(train_loss / s))


"""Tests the RBM.
Uses average distane method. Another way to evaluate RBM would be to use RMSE (root mean square errors).
RMSE is calculated as the root of the mean of the squared differences between the predictions and the targets.

Training phase:

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
Test phase:

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))

Using RMSE, RBM would obtain an error around 0.46. Although it looks similar, one must not confuse RMSE and the 
Average Distance. A RMSE of 0.46 doesn’t mean that the average distance between the prediction and the ground 
truth is 0.46. In random mode RMSE would end up around 0.72. An error of 0.46 corresponds to 75% of 
successful prediction.

Code to check what % final average distance is:
import numpy as np
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> 0.75
np.mean(np.abs(u-v)) # -> 0.25"""

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('test loss: ' + str(test_loss / s))
