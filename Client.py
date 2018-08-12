import cv2
import numpy as np
import socket
import pickle

if __name__ == '__main__':
    client_socket = socket.socket()
    port = 9090
    server_address = '197.50.98.221'
    while True:
        client_socket.connect((server_address, port))
        pickle_data = client_socket.recv(4096)
        data = np.array(pickle.loads(pickle_data))
        cv2.imshow('Virtual Whiteboard', data)
