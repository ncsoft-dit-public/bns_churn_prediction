#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import numpy as np

NODE1 = 20
NODE2 = 20
NODE3 = 20
NODE4 = 20
BATCH = 1000

# CSV 파일을 읽어들임
data = np.loadtxt('data/bns_churn_detection_nan.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:4])
y_data = np.transpose(data[4:])

# 모델 구성을 위한 변수생성
global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망 모델 구성
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([4,NODE1],-1.,1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([NODE1,NODE2],-1.,1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_uniform([NODE2,NODE3],-1.,1.), name='W3')
    L3 = tf.nn.relu(tf.matmul(L2, W3))

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_uniform([NODE3,NODE4],-1.,1.), name='W4')
    L4 = tf.nn.relu(tf.matmul(L3, W4))

with tf.name_scope('output'):
    W5 = tf.Variable(tf.random_uniform([NODE4, 3],-1.,1.), name='W5')
    model = tf.matmul(L4, W5)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
    tf.summary.scalar('cost', cost)


# 체크포인트 존재 유무에 따라 텐서플로우의 세션을 초기화
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('ckpt/model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/logs', sess.graph)

# 학습 및 모델 저장
for step in range(BATCH):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    # 10번에 1번씩 결과 출력
    if (step + 1) % 50 == 0:
        print('Step: %d' % sess.run(global_step),
              'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))
saver.save(sess, 'ckpt/model/dnn.ckpt', global_step=global_step)

# 결과확인
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값', sess.run(prediction, feed_dict={X: x_data}))
print('실제값', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %d, %d, %d, %d, %d, %.2f%%' % (NODE1, NODE2, NODE3, NODE4, BATCH, sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})))
