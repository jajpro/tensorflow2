import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_data = np.array([[1, 2, 0], [5, 4, 3], [1, 2, -1], [3, 1, 0], [2, 4, 2],
                   [4, 1, 2], [-1, 3, 2], [4, 3, 3], [0, 2, 6], [2, 2, 1],
                   [1, -2, -2], [0, 1, 3], [1, 1, 3], [0, 1, 4], [2, 3, 3]])
t_data = np.array([-4, 4, -6, 3, -4,
                   9, -7, 5, 6, 0,
                   4, 3, 5, 5, 1])
# 2a-3b+2c+0 = t_data

print('x_data.shape = ', x_data.shape, ', t_data.shape = ', t_data.shape)

# 모델구축
model = Sequential()
model.add(Dense(1, input_shape=(3, ), activation='linear'))  # 입력층 3개, 활성화 함수 linear regression

# 모델 컴파일
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')    # Stochastic Gradient Descent, mse 평균제곱오차
model.summary()  # 학습해야할 Param 4개 (w1, w2, w3, b)

# 모델학습
hist = model.fit(x_data, t_data, epochs=1000)

# 모델평가 및 예측
test_data = [[5, 5, 0], [2, 3, 1], [-1, 0, -1], [10, 5, 2], [4, -1, -2]]
ret_val = [2*data[0]-3*data[1]+2*data[2] for data in test_data]
prediction_val = model.predict(np.array(test_data))
print(prediction_val)
print("====================")
print(ret_val)

print(model.input)  # 모델 입력
print(model.output)  # 모델 출력
print(model.weights)  # 모델 가중치 & 바이어스

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

# 손실함수는 hist.history['loss'] 자동 저장됨
plt.plot(hist.history['loss'], label='train loss')
plt.legend(loc='best')
plt.show()


