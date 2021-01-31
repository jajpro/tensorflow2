import tensorflow as tf
import numpy as np

print(tf.__version__)

"""
Constant & Variable & placeholder
Tensorflow 1.x 에서는 계산 그래프를 선언하고, 세션을 통해 Tensor 를 주고 받으며 계산하는 구조였으나
Tensorflow 2.x 에서는 자동으로 Eager Execution(즉시 실행 모드) 적용되어 그래프와 세션 만들지 않아도 Tensor 값 계산
"""
print("Constant ==============================")

a = tf.constant(10)
b = tf.constant(20)
c = a+b
print("type(c) : ", type(c))
print("c : ", c)    # 저장된 값이 아닌 현재 정의되어 있는 노드의 상태 출력

"""
1.x 는 아래처럼 해야 연산 실행
with tf.Session() as sess:
    print(sess.run(c))
2.x 에서는 Session 만들지 않아도됨
"""

d = (a+b).numpy()   # numpy 값 반환. Eager Execution. 실행 순간 연산 실행
print("type(d) : ", type(d))
print("d : ", d)

d_numpy_to_tensor = tf.convert_to_tensor(d)  # numpy 값을 tensor 값으로 반환
print("type(d_numpy_to_tensor)", type(d_numpy_to_tensor))
print("d_numpy_to_tensor", d_numpy_to_tensor)

print("Variable ==============================")

W = tf.Variable(tf.random.normal([1]))  # 가우시안 분포

"""
1.x 
with tf.Session() as sess:
    # 세션 내에서 변수 W 초기화 코드 실행 필요
    sess.run(tf.global_variables_initializer())
    for step in range(2):
        W =W + 1.0
        print("step = ", step, ", W = ", sess.run(W))
"""

print("initial W = ", W.numpy())

for step in range(2):
    W = W + 1.0
    print("step = ", step, ", W = ", W.numpy())

print("placeholder ==============================")

"""
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
def tensor_sum(x, y):
    return x+y
result = tensor_sum(a, b)
with tf.Session() as sess:
    print(sess.run(result, feed_dict={a: [1.0], b: [3.0]}))  # placeholder 노드에 삽입
"""

a = tf.constant(1.0)
b = tf.constant(3.0)


def tensor_sum(x, y):
    return x+y


result = tensor_sum(a, b)
print(result)
print(type(result))
print(result.numpy())   # Eager Execution

