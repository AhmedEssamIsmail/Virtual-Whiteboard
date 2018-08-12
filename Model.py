import tensorflow as tf
import numpy as np
import cv2


def initialize_parameters():
    parameters = {}

    parameters["W1"] = tf.get_variable('W1', [4, 4, 1, 8],
                                       initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W2"] = tf.get_variable('W2', [4, 4, 8, 16],
                                       initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W3"] = tf.get_variable('W3', [2, 2, 16, 32],
                                       initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W4"] = tf.get_variable('W4', [32, 14], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters["b1"] = tf.get_variable('b1', [8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b2"] = tf.get_variable('b2', [16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b3"] = tf.get_variable('b3', [32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b4"] = tf.get_variable('b4', [14], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    return parameters


def forward_prop(X, parameters):
    Z1 = tf.nn.bias_add(
        tf.nn.conv2d(X, parameters["W1"], strides=[1, 1, 1, 1], padding='SAME'),
        parameters["b1"])

    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

    Z2 = tf.nn.bias_add(
        tf.nn.conv2d(P1, parameters["W2"], strides=[1, 1, 1, 1], padding="SAME"),
        parameters["b2"])

    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    Z3 = tf.nn.bias_add(
        tf.nn.conv2d(P2, parameters["W3"], strides=[1, 1, 1, 1], padding="SAME"),
        parameters["b3"])

    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flatten = tf.contrib.layers.flatten(P3)
    FC = tf.nn.bias_add(tf.matmul(flatten, parameters['W4']), parameters['b4'])

    return FC


def classify(image, sess, parameters):
    x = tf.cast(tf.reshape(image, [1, 50, 50, 1]), tf.float32)
    FC = forward_prop(x, parameters)
    z = sess.run(FC)
    return int(np.argmax(z))


def predict(frame, box, sess, parameters):
    cropped = frame[box[1]:box[1] + box[3] - box[1], box[0]:box[0] + box[2] - box[0]]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped = cv2.resize(cropped, (50, 50), cv2.INTER_AREA)
    cropped = tf.reshape(cropped, [1, 50, 50, 1])
    classno = classify(cropped, sess, parameters)
    return classno


def switchcase(classno):
    if classno >= 0 and classno <= 9:
        return str(classno)
    if classno == 10:
        return 'End'
    if classno == 11:
        return 'Hold'
    if classno == 12:
        return '+'
    if classno == 13:
        return '-'
