import cv2
import numpy as np
import Board
import Rubber
import Utility
import socket
import tensorflow as tf
import Model
import time
import copy
import pickle

if __name__ == '__main__':
    '''
    serverSocket = socket.socket()
    port = 9090
    address = socket.gethostname()
    serverSocket.bind(('', port))
    serverSocket.listen(1)
    Time = int(time.time())
    '''
    color = (0, 0, 0)
    penThickness = 5
    rubberThickness = 3
    inwardDist = 50
    firstP = (0, 0)
    state = 0  # (0)Draw, (1)Rubber, (2)Calculator
    equation = ""
    savedEquation = ""
    lastToken = "start"
    equationFlag = False
    Start = False
    Timer = 0
    LastSec = 0
    handCascade = cv2.CascadeClassifier("aGest.xml")
    palmCascade = cv2.CascadeClassifier("palm.xml")
    video_capture = cv2.VideoCapture(0)  # change the value for mobile connection 'http://192.168.1._:4747/mjpegfeed'
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    board = np.ones([int(height), int(width), 3])
    with tf.Session() as sess:
        parameter = Model.initialize_parameters()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('Model/'))
        # Start
        while True:
            # conSocket, client_address = serverSocket.accept()
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            predict_frame = copy.deepcopy(frame)
            Box1, Box2, Box3 = Utility.DrawBoxes(frame)
            _,_=Utility.DetectHandPos(frame, palmCascade, color = (255,0,0))
            cv2.waitKey(10)
            if state == 2:
                #calculator is disable for future work
                state = 0
                continue
                # erase those two lines to enable it
                cv2.putText(frame, equation, (int(width / 4), int(height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                cv2.imshow('Video', frame)
                CurrSec = int(time.time())
                if Time < CurrSec != LastSec:
                    Timer += 1
                    LastSec = int(time.time())
                    print(Timer)
                if Timer >= 3:
                    Timer = 0
                    Time = int(time.time())
                    classno = Model.predict(predict_frame, Box3, sess, parameter)
                    token = Model.switchcase(classno)
                else:
                    continue
                if token == "End":
                    savedEquation = equation
                    state = 0
                    equationFlag = True
                    equation = ""
                    lastToken = 'Start'
                    Start = False
                    continue
                if token == "Hold" and Start == False:
                    Start = True
                    lastToken = "Hold"
                    continue
                if token == lastToken:
                    continue
                else:
                    lastToken = token
                if token != "Hold":
                    equation += token
            else:
                point, frame = Utility.DetectHandPos(frame, handCascade)
                if point is None:
                    cv2.imshow('Video', frame)
                    continue
                state = Utility.SwitchState(Box1, Box2, Box3, point, state)
                if state == 0:
                    if equationFlag:
                        Utility.positionEquation(board, equation, point[0], point[1])
                        equationFlag = False
                        continue
                    if firstP == (0, 0):
                        firstP = point
                    else:
                        secondP = point
                        Board.draw(board, firstP, secondP, color, penThickness, inwardDist)
                        firstP = secondP
                elif state == 1:
                    Rubber.Erase(board, point, rubberThickness)

                    # if client_address:
                    #    pickledata = pickle.dumps(board)
                    #    conSocket.send(pickledata)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
