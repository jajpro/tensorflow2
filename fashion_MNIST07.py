import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

print('\n train shape = ', x_train.shape,
      ', train label shape = ', t_train.shape)
print(' test shape = ', x_test.shape,
      ', test label shape =', t_test.shape)

print('\n train label = ', t_train)  # 학습데이터 정답 출력
print(' test label  = ', t_test)     # 테스트 데이터 정답 출력

plt.figure(figsize=(6, 6))
for index in range(25):
    plt.subplot(5, 5, index+1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
plt.show()

# 학습데이터 정답 분포 확인
label_distribution = np.zeros(10)
for idx in range(len(t_train)):
    label = int(t_train[idx])
    label_distribution[label] = label_distribution[label] + 1
print(label_distribution)

plt.title('train label distribution')
plt.grid()
plt.xlabel('label')
plt.hist(t_train, bins=10, rwidth=0.8)
plt.show()

# 학습 데이터 / 테스트 데이터 정규화 (Normalization) : 픽셀 컬러정보가 255임
x_train = (x_train - 0.0) / (255.0 - 0.0)
x_test = (x_test - 0.0) / (255.0 - 0.0)

# 정답 데이터 원핫 인코딩 수행 안함. 10진수 정답 바로 사용

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))    # 28x28 크기 2차원 이미지를 784개의 1차원 백터로 변환
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy'\
              , metrics=['accuracy'])   # 원핫 인코딩이 아니므로 sparse_categorical_crossentropy 사용
model.summary()

hist = model.fit(x_train, t_train, epochs=30, validation_split=0.3)

model.evaluate(x_test, t_test)

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')
plt.show()

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 6))
predicted_value = model.predict(x_test)
cm = confusion_matrix(t_test,   # 테스트 데이터 정답 10진수
                      np.argmax(predicted_value, axis=-1))  # 예측값 (softmax 를 통해 10개 노드로 출력) 중에 가장 큰 것 argmax 로 추출
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


print(cm)
print('\n')
for i in range(10):
    print(('label = %d\t(%d/%d)\taccuracy = %.3f') % (i, np.max(cm[i]), np.sum(cm[i]), np.max(cm[i])/np.sum(cm[i])))

# 정답 및 예측 값 분포 확인
label_distribution = np.zeros(10)
prediction_distribution = np.zeros(10)

print(predicted_value.shape)
for idx in range(len(t_test)):
    label = int(t_test[idx])
    label_distribution[label] = label_distribution[label] + 1
    prediction = int(np.argmax(predicted_value[idx]))
    prediction_distribution[prediction] = prediction_distribution[prediction] + 1
print(label_distribution)
print(prediction_distribution)
