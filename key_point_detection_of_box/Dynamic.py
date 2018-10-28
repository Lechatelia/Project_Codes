import numpy as np
import cv2

def Generate_Route(picture):

    img = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    cost = np.zeros(img.shape)
    cost_temp = 0
    position = np.zeros(img.shape)

    cost[:, 0] = img[:, 0]

    for co in range(img.shape[1] - 1):
        col = co + 1
        for row in range(img.shape[0]):
            cost_temp_temp = np.zeros(img.shape[0])
            for row_pre in range(img.shape[0]):
                cost_temp_temp[row_pre] = cost[row_pre, col - 1] - 10 * (row - row_pre) ** 2
            max_index = np.argmax(cost_temp_temp)
            max_value = cost_temp_temp[max_index]
            position[row, col] = max_index
            cost[row, col] = max_value + 2 * (img[row, col])

    route = np.zeros(img.shape)
    max_index = np.argmax(cost[:, img.shape[1] - 1])
    now_row = max_index
    for col in range(img.shape[1] - 1, 0, -1):
        route[int(now_row), col] = 255
        now_row = position[int(now_row), col]

    return route

