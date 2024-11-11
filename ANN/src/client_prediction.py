import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from main import ann

dataset = pd.read_csv('Assignment_Data.csv')
x = dataset.iloc[:, 3:-1].values

# Encoding the "Gender" column (index 2)
le = LabelEncoder()
x[:, 5] = le.transform(x[:, 5])

# One Hot Encoding the "Geography" column (index 0)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
x_new = np.array(ct.transform(x))

# Scaling the features using the same scaler used during training
#sc = StandardScaler()
#x_new = sc.fit_transform(x_new)

# Reshape x_new to match the input shape expected by the model (if necessary)
#x_new = np.reshape(x_new, (1, -1))

# Predict the outcome
prediction = ann.predict(x_new)

# The model will output a probability, so we round it to get the binary outcome
result = (prediction > 0.5).astype(int)
print(f'The customer will {"leave" if result[0][0] == 1 else "not leave"} the bank.')
