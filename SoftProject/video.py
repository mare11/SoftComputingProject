import random

import cv2
import math
import numpy as np

import image
from network import model


class Number:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frames = 0
        self.passed_blue = False
        self.passed_green = False


sum = 0
regions = []

crossed_blue = 0
crossed_green = 0


def load_video(path):
    global sum, crossed_blue, crossed_green
    sum = 0
    crossed_blue = 0
    crossed_green = 0

    frame_num = 0
    frames = []
    cap = cv2.VideoCapture(path)
    cap.set(0, frame_num)

    blue_lines = None
    green_lines = None

    while True:
        frame_num += 1
        ret_val, frame = cap.read()

        if not ret_val:
            break

        frames.append(frame)

        # mask borders
        low_blue = np.array([120, 0, 0])
        up_blue = np.array([255, 100, 50])

        low_green = np.array([0, 120, 0])
        up_green = np.array([50, 255, 20])

        if frame_num == 1:
            blue_edges = image.erode(image.dilate(mask_edges(frame, low_blue, up_blue)))
            green_edges = image.erode(image.dilate(mask_edges(frame, low_green, up_green)))

            blue_lines = get_lines(blue_edges)
            green_lines = get_lines(green_edges)

        # select regions of interest
        test = image.invert(image.image_bin(image.image_gray(frame)))
        test_bin = image.invert(image.dilate(image.erode(test)))
        frame, test_numbers = image.select_roi(frame, test_bin)

        if blue_lines is not None and green_lines is not None:
            check_lines(frame, blue_lines[0][0], test_numbers, 'BLUE')
            check_lines(frame, green_lines[0][0], test_numbers, 'GREEN')
        else:
            print('NO LINE!')

    #     cv2.imshow('frame', frame)
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    #
    # print('BLUE CROSSED:', crossed_blue)
    # print('GREEN CROSSED:', crossed_green)

    return sum


def random_number():
    return random.randint(1, 9)


def mask_edges(frame, lower, upper):
    mask = cv2.inRange(frame, lower, upper)
    return cv2.Canny(mask, 75, 150)


def get_lines(edges):
    return cv2.HoughLinesP(edges, 1, np.pi / 180, 35, maxLineGap=250)


def check_lines(frame, line, numbers, color):
    global regions, crossed_blue, crossed_green
    x1, y1, x2, y2 = line
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    coefficients = np.polyfit((x1, x2), (y1, y2), 1)
    a = coefficients[0]  # k
    b = coefficients[1]  # n

    # disappeared regions check
    # -------------------------
    for i, r in enumerate(regions):
        if r.frames >= 150 and len(regions) > i:
            del regions[i]
            # print('DELETED')
    # -------------------------

    used_regions = []
    numbers = np.array(numbers)
    for x in numbers:
        x_bot_new = x[1][0] + x[1][2]  # x + width
        y_bot_new = x[1][1] + x[1][3]  # y + height

        center_x_new = (x[1][0] + x_bot_new) / 2
        center_y_new = (x[1][1] + y_bot_new) / 2

        used = False
        for ind, r in enumerate(regions):

            x_diff = r.x - center_x_new
            y_diff = r.y - center_y_new
            dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
            if dist < 50:
                r.x = center_x_new
                r.y = center_y_new
                r.frames = 0
                used = True
                used_regions.append(ind)

                dist_line = y_bot_new - (a * x_bot_new + b)
                if abs(dist_line) < 6:
                    if x_bot_new >= x1 and x_bot_new <= x2 and y_bot_new <= y1 and y_bot_new >= y2:
                        if color == 'BLUE' and not r.passed_blue:
                            r.passed_blue = True
                            # print('PROSAO', color)
                            predict_number(x[0], color)
                            crossed_blue += 1
                        elif color == 'GREEN' and not r.passed_green:
                            r.passed_green = True
                            # print('PROSAO', color)
                            predict_number(x[0], color)
                            crossed_green += 1

        if not used and len(numbers) > len(regions):
            n = Number(center_x_new, center_y_new)
            regions.append(n)
            # print('NEW NUMBER!')

    unused_regions = [i for i, r in enumerate(regions) if i not in used_regions]
    for i in unused_regions:
        regions[i].frames += 1


def predict_number(input_image, color):
    global sum
    input_image = image.invert(input_image)
    pred = model.predict(input_image.reshape(1, 28, 28)).argmax()
    # pred = random_number()
    if color == 'BLUE':
        sum += pred
    else:
        sum -= pred
    # print('*****Sum: ', sum, '*****')
