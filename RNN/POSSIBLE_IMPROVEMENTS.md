Some different ways to improve the RNN model:

1. Getting more training data: model is trained on 5 years of the Google stock price date (the past), but it would be even better to train it on the past 10 years.

2. Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because number of 60 timesteps (3 months) is chosen. Could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).

3. Adding some other indicators: if the stock price of some other companies might be correlated to Googles stock price, could try and add this other stock price as a new indicator in the training data.

4. Adding more LSTM layers: RNN was built with four LSTM layers, but could try with even more.

5. Adding more neurones in the LSTM layers: the fact that a high number of neurones is required in the LSTM layers to respond better to the complexity of the problem is highlighted, and 50 neurones are included in each of the 4 LSTM layers. Could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.