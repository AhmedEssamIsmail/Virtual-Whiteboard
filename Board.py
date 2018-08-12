import cv2 as cv
import numpy as np


def draw(board, first_point, second_point, color, thickness, inwardDist):
    dist = np.sqrt(pow((first_point[0] - second_point[0]), 2) + pow((first_point[1] - second_point[1]), 2))
    if dist < inwardDist:
        cv.line(board, first_point, second_point, color, thickness)

    cv.imshow('Virtual Whiteboard', board)
