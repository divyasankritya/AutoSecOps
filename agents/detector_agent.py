import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Load log data
with open('/Users/anandsankritya/Desktop/PRO/AutoSecOps/logs/sample_intrusions.json', 'r') as f:
    logs = json.load(f)

df = pd.DataFrame(logs)
print("DataFrame Head")
print(df.head())

# Encode Categorical columns to number
label_encoders = {}
for col in ['user', 'action', 'status', 'severity']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Encoded Dataframe")
print(df[['user', 'action', 'status', 'severity']])

# # Select features
features = df[['user', 'action', 'status', 'severity']].values

# Normalize numerical features (user, action, status, severity) to 0-1 range
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print("Features Scaled")
print(features_scaled)


train_data = features_scaled[:3]
test_data = features_scaled

#  Build the AutoEncoder Model
# The autoencoder tries to compress and then reconstruct the input.
# The last layer has the same size as the input, so it can try to recreate it.

input_dim = features_scaled.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])


autoencoder.compile(optimizer='adam', loss='mse')

# Train the Auntoencoder The model learns to reconstruct normal data.

autoencoder.fit(train_data, train_data, epochs=100, batch_size=1, verbose=0)


# For each log entry, we see how well the autoencoder can reconstruct it.
# High error means the entry is different from what the model learned (potential anomaly).


reconstructions = autoencoder.predict(test_data)
mse = np.mean(np.square(test_data - reconstructions), axis=1) 
print("Reconstruction errors : ", mse)

# # Set threshold as max error seen in training data

threshold = np.max(np.mean(np.square(train_data - autoencoder.predict(train_data)), axis=1))
print("Anomaly threshold : ", threshold)
df['reconstruction_error'] = mse

# Any entry with error above the threshold is flagged as an anomaly.
df['anomaly'] = df['reconstruction_error'] > threshold


print("Anomalous log entries (potential attacks):")
print(df[df['anomaly']])