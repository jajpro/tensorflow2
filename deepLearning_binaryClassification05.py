import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).astype('float32')   # 입력: 공부시간
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).astype('float32')  # 정답: Fail/Pass

print(x_data.shape)

model = Sequential()
model.add(Dense(8, input_shape=(1, ), activation='sigmoid'))    # 은닉층 노드 8개, 입력층 노드 1개. w:8, b:8 -> Param#: 16
model.add(Dense(1, activation='sigmoid'))   # w:8, b:1 -> Param#: 9
# 학습시켜야할 Total Param#: 25



model.compile(optimizer=SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, t_data, epochs=500)

test_data = np.array([0.5, 3.0, 3.5, 11.0, 13.0, 31.0])
sigmoid_value = model.predict(test_data)

logical_value = tf.cast(sigmoid_value > 0.5, dtype=tf.float32)

for i in range(len(test_data)):
    print(test_data[i],
          sigmoid_value[i],
          logical_value.numpy()[i])

model.weights

