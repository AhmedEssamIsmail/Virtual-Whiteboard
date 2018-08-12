import cv2


def DetectHandPos(frame, handCascade, color=(0, 255, 0) ):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = handCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
    # print(hands)
    # Draw a rectangle around the hands
    for (x, y, w, h) in hands:
        a = int(x + (h / 2))
        b = int(y + (w / 2))
        cv2.rectangle(frame, (a, b), (a + 1, b + 1), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color , 2)
        return (a, b), frame
    return None, frame


def DrawBoxes(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontSize = 0.8
    offset = 25
    x, y, z = frame.shape
    percX = int(0.25 * y)
    percY = int(x / 3)
    x1 = y - percX
    y1 = 0
    x2 = y
    y2 = percY
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # box1
    cv2.putText(frame, 'Pen', (x1, y1 + offset), font, fontSize, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y2), (x2, y2 + percY), (0, 255, 0), 2)  # box2
    cv2.putText(frame, 'Rubber', (x1, y2 + offset), font, fontSize, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y2 + percY), (x2, y2 + 2 * percY), (0, 255, 0), 2)  # box3
    cv2.putText(frame, 'Calculator', (x1, y2 + percY + offset), font, fontSize, (0, 0, 255), 2, cv2.LINE_AA)

    return (x1, y1, x2, y2), (x1, y2, x2, y2 + percY), (x1, y2 + percY, x2, y2 + 2 * percY)


def SwitchState(Box1, Box2, Box3, point, state):
    if Box1[0] < point[0] and Box1[1] < point[1] and Box1[2] > point[0] and Box1[3] > point[1]:
        return 0
    if Box2[0] < point[0] and Box2[1] < point[1] and Box2[2] > point[0] and Box2[3] > point[1]:
        return 1
    if Box3[0] < point[0] and Box3[1] < point[1] and Box3[2] > point[0] and Box3[3] > point[1]:
        return 2
    return state


def positionEquation(board, equation, x, y):
    res = evaluateEquation(equation)
    cv2.putText(board, equation + "=" + str(res), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv2.imshow("Virtual Whiteboard", board)


def evaluateEquation(string):
    result = 0
    current_sign = "+"
    i = 0
    while True:
        if i >= len(string):
            break
        if string[i].isdigit():
            temp = ""
            while i < len(string) and string[i].isdigit():
                temp += string[i]
                i += 1
            if current_sign == "+":
                result += int(temp)
            elif current_sign == "-":
                result -= int(temp)
        if i < len(string) and string[i] == '+':
            current_sign = "+"
        if i < len(string) and string[i] == '-':
            current_sign = "-"
        i += 1
    return result
