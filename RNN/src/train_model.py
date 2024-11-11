import joblib
import keras
import numpy as np
import pandas as pd
from keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from sklearn.preprocessing import MinMaxScaler

"""Recurrent neural network for predicting Google stock price up/down trend"""

"""Part 1. Data preprocessing
Imports the data as a dataframe and splits the data into training and test sets.
"""
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

"""Feature scaling.
Whenever building an RNN and there is a sigmoid function as the activation function in the output layer, 
apply normalisation over standardisation.
Normalisation:        xNorm = (x - min(x)) / (max(x) - min(x))
Standardisation:      xStandard = (x - mean(x)) / standard deviation(x)

Uses MinMaxScaler from sklearn.preprocessing.
"""
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


"""
Creating a data structure with 60 time-steps and 1 output.
60 corresponds to 3 months. Using the wrong amount could lead to overfitting.
"""
X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

"""Reshaping the data.
Adding a new dimension to the data, representing another indicator to use for up/down trend prediction.
"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


"""Part 2. Building the RNN
Initializes the RNN"""
regressor = Sequential()

"""Adds the first LSTM layer to the RNN and some Dropout regularisation.
Units - nr of LSTM cells. 
return sequences - True if building a stacked RNN, otherwise False.
input shape - in 3D, corresponding to observations, timestamps and indicators. In LSTM only last two are included.
            The first one will be automatically taken into account.
"""
regressor.add(Input(shape=(X_train.shape[1], 1,)))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

"""Adds the second LSTM layer to the RNN and some Dropout regularisation."""
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

"""Adds the third LSTM layer to the RNN and some Dropout regularisation."""
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

"""Adds the fourth LSTM layer to the RNN and some Dropout regularisation."""
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

"""Adds the output layer"""
regressor.add(Dense(units=1))

"""Compiles the RNN"""
regressor.compile(optimizer='adam', loss='mean_squared_error')

"""Model Summary"""
# regressor.summary()


def fit_model():
    """Fits the RNN to the training set"""
    regressor.fit(X_train, Y_train, epochs=100, batch_size=32)

    """Save the RNN model."""
    keras.models.save_model(regressor, 'stock_trend_trained_model.h5')
    joblib.dump(sc, 'scaler.save')


def get_dataset_train():
    return dataset_train


if __name__ == '__main__':
    fit_model()
