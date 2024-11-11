"""Part 3. Making the predictions and visualizing the results.
Does not react to fast non-linear changes (spikes).
Brownian motion (Wiener process) a mathematical concept in finance
 - stock price future state is independent of past changes.
 """
import joblib
import numpy as np
import pandas as pd
from keras.src.saving import load_model
from matplotlib import pyplot as plt

import train_model

"""Gets the real stock price of a time period.
Create a dataframe.
Import the test set.
"""
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

"""Gets the predicted stock price of a time period
Model is trained on 60 previous stock prices. To predict the real stock price of each day, previous 60 days are needed.

In order to get the previous 60 days, both training set and test set are needed, because some are from training data,
some are from real data. This is achieved by concatenating the sets.

The original dataset's are concatenated and then scaled. 
"""
dataset_total = pd.concat((train_model.get_dataset_train()['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)

# Load the saved scaler
sc = joblib.load('scaler.save')
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

regressor = load_model('stock_trend_trained_model.h5')
predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = train_model.get_sc().predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

"""Visualizes the results"""
plt.plot(real_stock_price, color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
