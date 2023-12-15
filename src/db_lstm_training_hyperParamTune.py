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

start_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
print('start_time > ' + start_time)

# Create a connection to your database
engine = create_engine('mysql://root:0000@localhost:3306/eeg_state')

# Step 1: Load the data
# eeg_data = pd.read_sql_query("SELECT * FROM eeg_data ORDER BY time DESC", engine)
query = """
    SELECT
        time,
        space_id, 
        person_id, 
        state, 
        ch1_value, 
        ch2_value, 
        ch3_value, 
        ch4_value
    FROM 
        (
            SELECT 
                time,
                space_id, 
                person_id, 
                state, 
                ch1_value, 
                ch2_value, 
                ch3_value, 
                ch4_value,
                ROW_NUMBER() OVER (PARTITION BY space_id, person_id, state ORDER BY time DESC) AS row_num
            FROM 
                eeg_data
        ) AS data
    WHERE 
        row_num <= {N}
    ORDER BY 
        state, time
"""
resolution = 256
learning_size = 256
N = resolution * 56  # the number of records for each group
eeg_data = pd.read_sql_query(query.format(N=N), engine)


# Step 2: Group by space_id and person_id, and get the latest n rows
# eeg_data = eeg_data.groupby(['space_id', 'person_id', 'state']).head(N)

# Prepare data for LSTM
# Convert ch_values columns to a list of values
eeg_data['ch_values'] = eeg_data[['ch1_value', 'ch2_value', 'ch3_value', 'ch4_value']].apply(lambda x: x.values.tolist(), axis=1)

# print(eeg_data)
# Now, we have our time series data for each 'ch_values'. We groupby 'space_id', 'person_id', 'state' and convert our time series data for each group into a list.
eeg_grouped = eeg_data.groupby(['space_id', 'person_id', 'state'])['ch_values'].apply(list)
X = np.array([np.array(x) for x in eeg_grouped])
print(X[1].shape)

# # Prepare target variable
# y = np.unique(eeg_data.groupby(['space_id', 'person_id', 'state']).first().reset_index()['state'].values)
y = np.array([0,1,2,3,4])
print(y)
# y = to_categorical(y)  # One-hot encoding


# print(X)
# print('y shape: ', y)
# # print('X shape: ', len(X[4]))
# # print('X shape: ', X[4].shape)

# # # Prepare data for LSTM
# # # Convert ch_values columns to a list of values
# # X = eeg_data.groupby(['space_id', 'person_id', 'state'])[['ch1_value', 'ch2_value', 'ch3_value', 'ch4_value']].apply(lambda x: x.values.tolist())
# # y = eeg_data.groupby(['space_id', 'person_id'])['state'].apply(lambda x: x.values)

# # print('X: ', X)
# # print('y: ', y)

# # # Convert X to 3D array (samples, timesteps, features) for LSTM
# print(X[4].shape)
X = np.array(X.tolist()).reshape((-1, 256, 4))
print(X.shape)
y = np.repeat(y, X.shape[0]/5, axis=0)
y = to_categorical(y)  # One-hot encoding
# X = X.reshape((X.shape[0], X.shape[2], X.shape[3]))

# print(y)
# print('X: ', X)

# # # Reshape X
# # num_samples = int(n / learning_size)
# # X = np.reshape(X, (num_samples, learning_size, X.shape[2]))

# # # Reshape y (if needed)
# # y = np.repeat(y, num_samples, axis=0)

# # print('X_reshaped: ', X.shape)
# # print('y_reshaped: ', y.shape)

# # Convert y to one-hot encoding
# le = LabelEncoder()
# y = le.fit_transform(y)
# y = to_categorical(y, num_classes=5)

# Define the number of folds
# n_splits = 2

# # Initialize the TimeSeriesSplit
# tscv = TimeSeriesSplit(n_splits=n_splits)

# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    # Perform training and evaluation on each fold

# Split into train and test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Step 4: LSTM Learning
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)): # Defining the search space for number of layers
        model.add(layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32), 
                              return_sequences=True if i < hp.Int('num_layers', 2, 20)-1 else False)) # Defining the search space for number of units
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # set the maximum number of trials (model configurations) to try
    executions_per_trial=1,  # the number of models to train per trial to robustly estimate the performance of a configuration
    directory='../models/LSTM/',
    project_name='hyperParameterTuning_bestModel' + "_" + start_time)

tuner.search_space_summary()

tuner.search(X_train, y_train,
             epochs=20,
             validation_data=(X_val, y_val))

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

end_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
print('end_time > ' + end_time)
# Step 5: Save the trained model
best_model.save("../models/LSTM/eeg2emotionstate_LSTM_hpt_best" + "_" + start_time + "_" + end_time)
