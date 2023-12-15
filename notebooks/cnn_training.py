

############### Import-Data #####################


#SAXVSM
import numpy as np
import os
import scipy.io

from scipy import signal

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data_root = '../EEG Data/'

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
channel_map = dict(zip(channel_names, channel_indices))

subjects = range(28, 35)
scores = {}
selected_channel_names = ['T7', 'AF3', 'AF4', 'T8']
select_indices = np.array([channel_names.index(element) for element in selected_channel_names])
# select_indices = channel_indices
spectrograms_ch = []
spectrograms_ch_mark = []
spectrograms = []
spectrograms_mark = []

for ch in select_indices:
    for subject_idx in subjects:
        states = get_data(f"eeg_record{subject_idx}.mat")

        tmp = states['focused'][:,ch].tolist()[:76800]
        if len(tmp) > 0: 
            spectrograms_ch_mark.append(1)
            # Generate spectrograms for each EEG channel
            frequencies, times, Sxx = signal.spectrogram(np.array(tmp), fs=128)
            Sxx = StandardScaler().fit_transform(Sxx)
            spectrograms_ch.append(Sxx)
        tmp = states['unfocused'][:,ch].tolist()[:76800]
        if len(tmp) > 0: 
            spectrograms_ch_mark.append(0)
            # Generate spectrograms for each EEG channel
            frequencies, times, Sxx = signal.spectrogram(np.array(tmp), fs=128)
            Sxx = StandardScaler().fit_transform(Sxx)
            spectrograms_ch.append(Sxx)
        tmp = states['drowsy'][:,ch].tolist()[:76800]
        if len(tmp) > 0: 
            spectrograms_ch_mark.append(-1)
            # Generate spectrograms for each EEG channel
            frequencies, times, Sxx = signal.spectrogram(np.array(tmp), fs=128)
            Sxx = StandardScaler().fit_transform(Sxx)
            spectrograms_ch.append(Sxx)
    spectrograms.append(spectrograms_ch)
    spectrograms_mark.append(spectrograms_ch_mark)

spectrograms = np.array(spectrograms)
spectrograms_mark = np.array(spectrograms_mark)

# train_y = spectrograms_mark[0:int(len(spectrograms_mark)*0.7)]
# test_y = spectrograms_mark[int(len(spectrograms_mark)*0.7):]
# train_x = spectrograms[0:int(len(spectrograms)*0.7)]
# test_x = spectrograms[int(len(spectrograms)*0.7):]

# train_x = np.array(train_x)
# train_y = np.array(train_y)

############### Training #####################

# 데이터 정규화
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# 데이터를 학습 및 테스트 세트로 분할
new_shape = (spectrograms.shape[0] * spectrograms.shape[1], Sxx.shape[0], Sxx.shape[1])
spectrograms = spectrograms.reshape(new_shape)
spectrograms_mark = spectrograms_mark.reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(spectrograms, spectrograms_mark, test_size=0.3)

# CNN에 입력으로 사용할 수 있는 4D 텐서로 변환
img_rows = Sxx.shape[0]
img_cols = Sxx.shape[1]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# 클래스 라벨을 이진 벡터로 변환
num_classes = 3
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print('start')

# 모델 구축:
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일:
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

# 모델 학습:
model.fit(X_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(X_test, y_test))

# 모델 평가:
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 모델 튜닝:
# 모델의 성능을 개선하기 위해 하이퍼파라미터를 조정하거나 모델 아키텍처를 변경하고, 이 과정을 반복합니다.
# 예를 들어, 계층의 수를 늘리거나 줄이거나, 컨볼루션 필터의 크기를 변경하거나, 드롭아웃 비율을 조정하거나, 
# 학습률을 변경하거나, 다른 최적화 알고리즘을 사용할 수 있습니다.

# 예측 결과 분석:
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 분류 보고서와 혼동 행렬 출력
print('Classification Report:')
print(classification_report(y_true, y_pred_classes))
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred_classes))


# ############### Pre-Processing #####################

# import matplotlib.pyplot as plt
# from scipy import signal
# import numpy as np

# # Assuming 'eeg_data' is your preprocessed EEG data and 'sampling_rate' is the sampling rate of your data
# # replace 'eeg_data' and 'sampling_rate' with your actual data and sampling rate
# eeg_data = your_preprocessed_eeg_data  
# sampling_rate = your_sampling_rate

# # Generate spectrograms for each EEG channel
# spectrograms = []

# for channel in eeg_data:
#     frequencies, times, Sxx = signal.spectrogram(channel, fs=sampling_rate)
#     spectrograms.append(Sxx)

# spectrograms = np.array(spectrograms)

# ############### Training #####################

# # 데이터 정규화
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # 데이터를 학습 및 테스트 세트로 분할
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # CNN에 입력으로 사용할 수 있는 4D 텐서로 변환
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# # 클래스 라벨을 이진 벡터로 변환
# num_classes = 3
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)

# # 모델 구축:
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # 모델 컴파일:
# model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

# # 모델 학습:
# model.fit(X_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(X_test, y_test))

# # 모델 평가:
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # 모델 튜닝:
# # 모델의 성능을 개선하기 위해 하이퍼파라미터를 조정하거나 모델 아키텍처를 변경하고, 이 과정을 반복합니다.
# # 예를 들어, 계층의 수를 늘리거나 줄이거나, 컨볼루션 필터의 크기를 변경하거나, 드롭아웃 비율을 조정하거나, 
# # 학습률을 변경하거나, 다른 최적화 알고리즘을 사용할 수 있습니다.

# # 예측 결과 분석:
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # 분류 보고서와 혼동 행렬 출력
# print('Classification Report:')
# print(classification_report(y_true, y_pred_classes))
# print('Confusion Matrix:')
# print(confusion_matrix(y_true, y_pred_classes))