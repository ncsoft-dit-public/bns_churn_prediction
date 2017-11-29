import tensorflow as tf
import numpy as np

# 학습용 데이터 생성
x_data = np.array([
    [1.9,1.8,0.2,0.2],
    [0.9,0.7,0.0,0.1],
    [-0.1,-0.1,-2.8,-0.1],
    [0.0,0.0,0.0,9.0]
])
# churn = [1,0,0], not_churn = [0,1,0], undefined = [0,0,1]
y_data = np.array([
    [1,0,0],
    [1,0,0],
    [0,1,0,],
    [0,0,1]
])

# 변수생성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([4,3], -1.0, 1.0)) # [4: 변수의 개수, 3: 카테고리의 개수]
b = tf.Variable(tf.zeros([3])) # 카테고리 별 bias

# 네트워크 및 모델 생성
L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)
model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로우의 세션을 초기화
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# 100번 학습 진행
for step in range(500):
    session.run(train_op, feed_dict={X: x_data, Y: y_data})
    # 10번에 1번씩 결과 출력
    if (step + 1) % 50 == 0:
        print(step + 1, session.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값', session.run(prediction, feed_dict={X: x_data}))
print('실제값', session.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f%%' % session.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
