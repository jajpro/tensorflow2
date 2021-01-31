import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from datetime import datetime


# Training Data 생성
try:

    loaded_data = np.loadtxt('./diabetes.csv', delimiter=',')

    # training data / test data 분리

    seperation_rate = 0.3  # 분리 비율
    test_data_num = int(len(loaded_data) * seperation_rate)

    np.random.shuffle(loaded_data)

    test_data = loaded_data[0:test_data_num]
    training_data = loaded_data[test_data_num:]

    # training_x_data / training_t__data 생성

    training_x_data = training_data[:, 0:-1]    # 모든 행에 대해 처음부터 마지막 열 전까지 slicing
    training_t_data = training_data[:, [-1]]    # 모든 행에 대해 마지막 열 추출

    # test_x_data / test_t__data 생성
    test_x_data = test_data[:, 0:-1]
    test_t_data = test_data[:, [-1]]

    print("loaded_data.shape = ", loaded_data.shape)
    print("training_x_data.shape = ", training_x_data.shape)
    print("training_t_data.shape = ", training_t_data.shape)

    print("test_x_data.shape = ", test_x_data.shape)
    print("test_t_data.shape = ", test_t_data.shape)

except Exception as err:

    print(str(err))

# 모델 구축 - Logistic Regression 을 keras 이용하여 생성

model = Sequential()

# 노드 1개인 출력층 생성
model.add(Dense(training_t_data.shape[1],   # 1(열)
                input_shape=(training_x_data.shape[1],),    # 8(열)
                activation='sigmoid'))

# 모델 컴파일 - 학습을 위한 optimizer, 손실함수 loss 정의

model.compile(optimizer=SGD(learning_rate=0.01),
              loss='binary_crossentropy',   # 결과가 0 또는 1이므로
              metrics=['accuracy'])

model.summary()

start_time = datetime.now()

# training data 로 부터 20% 비율로 validation data 생성 후 over fitting 확인
hist = model.fit(training_x_data, training_t_data, epochs=500, validation_split=0.2, verbose=2)

end_time = datetime.now()

print('\nElapsed Time => ', end_time - start_time)

model.evaluate(test_x_data, test_t_data)    # 기본 batch_size=32 인데, 전체 데이터 759개중 32개씩 가져오므로 759/32=23.7 로 24회 실행

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend(loc='best')

plt.show()


# train accuracy 와 validation accuracy 가 벌어지는 지점이 over fitting 이므로 epochs 조절
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()
