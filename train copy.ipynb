from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import csv
import os

# training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


DATA_PATH = os.path.join('MP_Data')  # Path for exported data, numpy arrays


#getting dataset values from csv
with open('datalabels.csv', newline='') as f:
    reader = csv.reader(f)
    data = [row[0] for row in reader]


actions = np.array(data)
no_sequences = 30  # thirty videos worth of data
sequence_length = 30  # each video is 30 frame in length

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []  # x data, y data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
print(x.shape)
y = to_categorical(labels).astype(int)
print(y)

# creating training partitions
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# training

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential() # instantiating the model
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

print(model.summary())

res = model.predict(x_test)
print(actions[np.argmax(res[0])])
print(actions[np.argmax(res[9])])

# save weights
model.save('action.h5')


#evaluation using confusion matrix
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(y_test).tolist()
ytrue = np.argmax(y_test).tolist()
yhat = np.argmax(yhat).tolist()
matrix = multilabel_confusion_matrix(ytrue, yhat)
print(matrix)
accuracy_score = accuracy_score(ytrue, yhat)
print(f"{accuracy_score= }")
