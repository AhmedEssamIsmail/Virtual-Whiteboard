import cv2


def neighbours(image, point, color=(1, 1, 1), thickness=5):
    sign = [1, -1]
    for i in range(thickness):
        for j in range(thickness):
            for a in sign:
                for b in sign:
                    new_point = (point[0] + (i * a), point[1] + (j * b))
                    if not (new_point[0] < 0 or new_point[1] < 0 or new_point[0] >= image.shape[0] or
                                    new_point[1] >= image.shape[1]):
                        image[new_point] = color


def Erase(board, point, thickness=5):
    point = (point[1], point[0])
    neighbours(board, point)
    neighbours(board, point, color=(0, 0, 1), thickness=thickness)
    cv2.imshow("Virtual Whiteboard", board)
    neighbours(board, point)
