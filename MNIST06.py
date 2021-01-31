import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_data()

print('\n train shape = ', x_train.shape,
      ', train label shape = ', t_train.shape)
print(' test shape = ', x_test.shape,
      ', test label shape =', t_test.shape)

print('\n train label = ', t_train)  # 학습데이터 정답 출력
print(' test label  = ', t_test)     # 테스트 데이터 정답 출력

# 25개 이미지 출력
plt.figure(figsize=(6, 6))

for index in range(25):
    plt.subplot(5, 5, index+1)  # 5행 5열
    plt.imshow(x_train[index], cmap='gray')  # 색상 gray
    plt.axis('off')  # 축 표시 제거
    plt.colorbar()
    # plt.title(str(t_train[index]))
plt.show()

# 학습데이터 정답 분포 확인 1
plt.title('train label distribution')
plt.grid()
plt.xlabel('label')
plt.hist(t_train, bins=10, rwidth=0.8)
plt.show()

# 학습데이터 정답 분포 확인 2
label_distribution = np.zeros(10)
for idx in range(len(t_train)):
    label = int(t_train[idx])
    label_distribution[label] = label_distribution[label] + 1
print(label_distribution)

# 학습 데이터 / 테스트 데이터 정규화 (Normalization) : 픽셀 컬러정보가 255임
x_train = (x_train - 0.0) / (255.0 - 0.0)
x_test = (x_test - 0.0) / (255.0 - 0.0)

# 정답 데이터 원핫 인코딩 (One-Hot Encoding)
t_train = tf.keras.utils.to_categorical(t_train, num_classes=10)
t_test = tf.keras.utils.to_categorical(t_test, num_classes=10)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
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

# 혼동행렬 Confusion matrix - 모델이 어떠한 숫자를 가장 혼란스러워하는지 파악 가능
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 6))
predicted_value = model.predict(x_test)
cm = confusion_matrix(np.argmax(t_test, axis=-1),
                      np.argmax(predicted_value, axis=-1))

sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print(cm)
print('\n')
for i in range(10):
    print(('label = %d\t(%d/%d)\taccuracy = %.3f') % (i, np.max(cm[i]), np.sum(cm[i]), np.max(cm[i])/np.sum(cm[i])))

# 정답 및 예측 값 분포 확인
label_distribution = np.zeros(10)
prediction_distribution = np.zeros(10)

for idx in range(len(t_test)):
    label = int(np.argmax(t_test[idx]))
    label_distribution[label] = label_distribution[label] + 1
    prediction = int(np.argmax(predicted_value[idx]))
    prediction_distribution[prediction] = prediction_distribution[prediction] + 1

print(label_distribution)
print(prediction_distribution)

