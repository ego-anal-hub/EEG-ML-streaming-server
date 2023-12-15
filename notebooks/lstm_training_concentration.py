import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras import models, layers, optimizers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from joblib import dump
import scipy.io
import os

data_root = '../data/concentration/'

def get_data(filename):
    mat = scipy.io.loadmat(os.path.join(data_root, filename))
    data = mat['o']['data'][0, 0]
    FS = mat['o']['sampFreq'][0][0][0][0]

    states = {
     'focused': data[:FS * 10 * 60, :],
      'unfocused': data[FS * 10 * 60:FS * 20 * 60, :],
      'drowsy': data[FS * 30 * 60:, :],
    }
    return states

channel_indices = np.array(range(3, 17))
channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# channel_names = ['AF3', 'F7', 'F3', 'T7', 'T8', 'F4', 'F8', 'AF4']
# SAXVSM(window_size=100) {3: 0.75, 4: 0.75, 5: 0.25, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.25, 11: 0.5, 12: 0.25, 13: 0.25, 14: 0.5, 15: 0.5, 16: 0.25}
# SAXVSM(window_size=300,overlapping=False,sublinear_tf=False, use_idf=False) {3: 0.5, 4: 0.5, 5: 0.75, 6: 0.5, 7: 0.5, 8: 0.25, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5}
# SAXVSM(window_size=64) {3: 0.5, 4: 0.25, 5: 0.25, 6: 0.75, 7: 0.5, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.5, 13: 0.25, 14: 0.5, 15: 0.5, 16: 0.25}
# SAXVSM(window_size=64,overlapping=False,sublinear_tf=False, use_idf=False) {3: 0.75, 4: 0.25, 5: 0.25, 6: 0.75, 7: 0.5, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.25, 12: 0.25, 13: 0.5, 14: 0.25, 15: 0.5, 16: 0.25}
# SAXVSM(window_size=32,overlapping=False,sublinear_tf=False, use_idf=False) {3: 0.5, 4: 0.75, 5: 0.75, 6: 0.5, 7: 0.5, 8: 0.75, 9: 0.75, 10: 0.75, 11: 0.75, 12: 0.25, 13: 0.25, 14: 0.25, 15: 1.0, 16: 0.75}
# SAXVSM(window_size=32) {3: 0.25, 4: 0.5, 5: 0.75, 6: 0.25, 7: 0.5, 8: 0.75, 9: 0.75, 10: 0.75, 11: 0.75, 12: 0.75, 13: 0.25, 14: 0.5, 15: 1.0, 16: 0.5}
# SAXVSM(window_size=256) {3: 0.75, 4: 0.5, 5: 0.75, 6: 0.5, 7: 0.5, 8: 0.25, 9: 0.5, 10: 0.25, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5}

# SAXVSM(window_size=32,overlapping=False,sublinear_tf=False, use_idf=False) {3: 0.25, 4: 0.625, 5: 0.625, 6: 0.375, 7: 0.25, 8: 0.625, 9: 0.625, 10: 0.625, 11: 0.625, 12: 0.25, 13: 0.5, 14: 0.5, 15: 0.375, 16: 0.625}
channel_map = dict(zip(channel_names, channel_indices))

subjects = range(10, 35)
# subjects = [28,35]
scores = {}
# select_indices = [5]
#select_indices = channel_indices
c1 = int(76800*0.1)
c2 = int(76800*0.9)
selected_channel_names = ['T7', 'AF3', 'AF4', 'T8']
select_indices = np.array([channel_names.index(element) for element in selected_channel_names])
X = []
y = []
for ch in select_indices:
    ch_data = []
    ch_mark = []

    for subject_idx in subjects:
        states = get_data(f"eeg_record{subject_idx}.mat")

        tmp = states['focused'][:,ch].tolist()[:76800]
        if len(tmp) == 76800:
            # tmp = tmp[:int(76800*0.8)]
            df = pd.Series(tmp)
            df = df.fillna(method='ffill')
            tmp = df.to_numpy()
            min_val = min(tmp)
            max_val = max(tmp)

            # Apply min-max normalization
            tmp = [(x - min_val) / (max_val - min_val) for x in tmp]
            ch_data.append(tmp)
            ch_mark.append(1)
        # tmp = [1 for i in range(len(states['focused'][:,ch].tolist()[:76800]))]
        # if len(tmp) > 0: 
        tmp = states['unfocused'][:,ch].tolist()[:76800]
        if len(tmp) == 76800: 
            # tmp = tmp[:int(76800*0.8)]
            df = pd.Series(tmp)
            df = df.fillna(method='ffill')
            tmp = df.to_numpy()
            min_val = min(tmp)
            max_val = max(tmp)

            # Apply min-max normalization
            tmp = [(x - min_val) / (max_val - min_val) for x in tmp]
            ch_data.append(tmp)
            ch_mark.append(0)
        # tmp = [0 for i in range(len(states['unfocused'][:,ch].tolist()[:76800]))]
        # if len(tmp) > 0: 
        tmp = states['drowsy'][:,ch].tolist()[:76800]
        if len(tmp) == 76800: 
            # tmp = tmp[:int(76800*0.8)]
            df = pd.Series(tmp)
            df = df.fillna(method='ffill')
            tmp = df.to_numpy()
            min_val = min(tmp)
            max_val = max(tmp)

            # Apply min-max normalization
            tmp = [(x - min_val) / (max_val - min_val) for x in tmp]
            ch_data.append(tmp)
            ch_mark.append(-1)
    X.append(ch_data)
    y.append(ch_mark)
        
# Step 2: Group by space_id and person_id, and get the latest n rows
resolution = 256
learning_size = 256
n = resolution * 5

# Convert X to 3D array (samples, timesteps, features) for LSTM
X = np.array(X)
X = X.reshape((X.shape[1], X.shape[2], X.shape[0]))

print('X: ', X.shape)

# Reshape X
# num_samples = int(n / learning_size)
# X = np.reshape(X, (num_samples, learning_size, X.shape[2]))
X = X.reshape((-1,256,4))

# # Reshape y (if needed)
y = np.array(y[0])
y = np.repeat(y, 300, axis=0)

print('X_reshaped: ', X.shape)
print('y_reshaped: ', y.shape)
print(y)

# # Convert y to one-hot encoding
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, num_classes=3)

# # Define the number of folds
# # n_splits = 2

# # # Initialize the TimeSeriesSplit
# # tscv = TimeSeriesSplit(n_splits=n_splits)

# # for train_index, test_index in tscv.split(X):
# #     X_train, X_test = X[train_index], X[test_index]
# #     y_train, y_test = y[train_index], y[test_index]
#     # Perform training and evaluation on each fold

# # Split into train and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)

# Split the temp data into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

print('X_train : ', X_train.shape)
print('y_train : ', y_train.shape)

# Step 4: LSTM Learning
# model = models.Sequential()
# model.add(layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(layers.Dense(y.shape[1], activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Fit the model
# model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, validation_data=(X_test, y_test))

model = models.Sequential()
model.add(layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dropout(0.2))  # 20% dropout
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dropout(0.2))  # 20% dropout
model.add(layers.Dense(100, activation='relu'))  # Add a dense layer
model.add(layers.Dense(y.shape[1], activation='softmax'))

# Use the Adam optimizer with a learning rate of 0.001
optimizer = optimizers.SGD(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and model checkpoint callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))