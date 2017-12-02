#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import numpy as np

#
# https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
#
# CSV 파일을 읽어들임
data = np.loadtxt('data/bns_churn_detection_nan.csv', delimiter=',', unpack=True, dtype='float32')

# 1. 일반적인 데이터 추출
x_data = np.transpose(data[0:4])
y_data = np.transpose(data[4:])

# 2. 미니배치 데이터 추출
dataset_length = len(data[1])

# 79% x=400, nodes=60, learning_rate=0.00005, keep_prob=0.6
x = 400
nodes = 60
learning_rate = 0.00005 # 학습률을 높이면 빠르게 학습은 되지만 local-maxima 에 빠지기 쉬워서 정확률의 편차가 커지더라 or Batch Normalization 적용

input = 4
batch_size = 100
batch_step = 8
total_batch = int(dataset_length/batch_size)

def exit(code=0):
    sys.exit(code)

# 모델 구성을 위한 변수생성
global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# 신경망 모델 구성
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([4,nodes],stddev=0.01), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
    L1 = tf.nn.dropout(L1, keep_prob)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([nodes,nodes],stddev=0.01), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([nodes,nodes],stddev=0.01), name='W3')
    L3 = tf.nn.relu(tf.matmul(L2, W3))
    L3 = tf.nn.dropout(L3, keep_prob)

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_normal([nodes,nodes],stddev=0.01), name='W4')
    L4 = tf.nn.relu(tf.matmul(L1, W2))
    L4 = tf.nn.dropout(L4, keep_prob)

with tf.name_scope('output'):
    W5 = tf.Variable(tf.random_normal([nodes, 3],stddev=0.01), name='W5')
    model = tf.matmul(L4, W5)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    # train_op = optimizer.minimize(cost, global_step=global_step)
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

# 총 5개의 학습집합 800 * 5 = 4000명
for epoch in range(x):
    total_cost = 0
    # 800개를 100 * 8개의 미니배치로 수행
    for step in range(total_batch):
        batch_mask = np.random.random_integers(0, dataset_length-1, batch_size)
        train_x = x_data[batch_mask]
        train_y = y_data[batch_mask]
        _, cost_val = sess.run([optimizer, cost],
                 feed_dict={X: train_x,
                            Y: train_y,
                            keep_prob: 0.6})
        # print('Step: %d/%d' % (step, sess.run(global_step)), 'Cost: %.3f' % sess.run(cost, feed_dict={X: train_x, Y: train_y}))
        summary = sess.run(merged, feed_dict={X: train_x,
                                              Y: train_y,
                                              keep_prob: 0.6})
        writer.add_summary(summary, global_step=sess.run(global_step))
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

# 결과확인
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
# print('예측값', sess.run(prediction, feed_dict={X: train_x}))
# print('실제값', sess.run(target, feed_dict={Y: train_y}))
saver.save(sess, 'ckpt/model/dnn.ckpt', global_step=global_step)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: nodes:%d, %.2f%%' % (nodes, sess.run(accuracy * 100,
                                                 feed_dict={X: train_x,
                                                            Y: train_y,
                                                            keep_prob: 1})))

