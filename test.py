import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2


def initialize_parameters():
    parameters = {}

    parameters["W1"] = tf.get_variable('W1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W2"] = tf.get_variable('W2', [4, 4, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W3"] = tf.get_variable('W3', [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["W4"] = tf.get_variable('W4', [32, 15], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters["b1"] = tf.get_variable('b1', [8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b2"] = tf.get_variable('b2', [16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b3"] = tf.get_variable('b3', [32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters["b4"] = tf.get_variable('b4', [15], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    return parameters


def read_image():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        x1 = frame.shape[1] - 200
        y1 = frame.shape[0] - 350
        x2 = x1 + 198
        y2 = y1 + 240
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            save = frame[y1:y1 + y2 - y1, x1:x1 + x2 - x1]
            s = save
            save = cv2.cvtColor(save, cv2.COLOR_BGR2GRAY)
            save = cv2.resize(save, (50, 50), cv2.INTER_AREA)
            cv2.imwrite("IMGg2.jpg", save)
            save = tf.reshape(save, [1, 50, 50, 1])
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return save, s


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


def predict(_X, _sess):
    sess = _sess
    par = initialize_parameters()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('model2/'))
    FC = forward_prop(_X, par)
    return sess.run(FC)


def classify(image, sess, par):
    x = tf.cast(tf.reshape(image, [1, 50, 50, 1]), tf.float32)
    FC = forward_prop(x, par)
    z = sess.run(FC)
    print(np.argmax(z))


if __name__ == '__main__':
    equ = ""
    tmp = 100
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        par = initialize_parameters()
        image, s = read_image()
        cv2.imshow("image", s)
        classify(image)
        cv2.waitKey()

        '''
        while True:

            image = read_image()

            currunt = go(image)

            if tmp == currunt:
                continue

            if currunt in range(0, 9):
                equ += str(currunt)
                tmp = currunt

            elif currunt == 10:
                break

            elif currunt == 11:
                temp = 100
                continue

            elif currunt == 13:
                equ += '+'
                tmp = currunt

            elif currunt == 14:
                equ += '-'
                tmp = currunt
 print(equ)

             '''
