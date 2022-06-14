import cv2 as cv
import sys
import copy
import pandas as pd
import numpy as np


def add_correct_cicle(image, list, window):
    image = cv.cvtColor(image, cv.CV_8UC1)
    if window == 1:
        center = (list[0], list[1])
        # circle center
        cv.circle(image, center, 1, (170, 178, 32), 3)
        # circle outline
        radius = list[2]
        cv.circle(image, center, radius, (154, 250, 0), 3)
    elif window == 2:
        center = (list[3], list[4])
        # circle center
        cv.circle(image, center, 1, (170, 178, 32), 3)
        # circle outline
        radius = list[5]
        cv.circle(image, center, radius, (154, 250, 0), 3)

    return image


def compute_square_vertices(x, y, radius):
    x = int(x)
    y = int(y)
    radius = int(radius)

    # x1 = x - radius
    # y1 = y + radius
    # x2 = x + radius
    # y2 = y - radius

    x1 = x - radius
    y1 = y - radius
    x2 = x + radius
    y2 = y + radius

    return int(x1), int(y1), int(x2), int(y2)


def compute_iou(square_circle, square_correct_circle):
    x_inter1 = max(square_circle[0], square_correct_circle[0])
    y_inter1 = max(square_circle[1], square_correct_circle[1])
    x_inter_2 = min(square_circle[2], square_correct_circle[2])
    y_inter_2 = min(square_circle[3], square_correct_circle[3])

    width_inter = abs(x_inter_2 - x_inter1)
    height_inter = abs(y_inter_2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(square_circle[2] - square_circle[0])
    height_box1 = abs(square_circle[3] - square_circle[1])
    width_box2 = abs(square_correct_circle[2] - square_correct_circle[0])
    height_box2 = abs(square_correct_circle[3] - square_correct_circle[1])

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter

    return area_inter / area_union


def refresh_image(hist, gaus_kerner, gaus_sigma, canny, houghAkumulator, houghMinVzCentKruznic,
                  param1, param2, houghMinPolKruznic, houghMaxPolKruznic):
    image = copy.copy(img)

    image = set_histogram(image, hist)

    image = set_gaus_rozm_jadro(image, gaus_kerner, gaus_sigma)

    image = set_gaus_rozm_sigma(image, gaus_sigma, gaus_kerner)

    image = set_canny(image, canny)

    is_houg = 1

    if is_houg == 1:
        result = set_hough(image, list[0:3], houghAkumulator, houghMinVzCentKruznic, param1, param2, houghMinPolKruznic,
                           houghMaxPolKruznic)
        image = result[0]

    image = add_correct_cicle(image, list, 1)
    cv.imshow(img_name, image)


def set_histogram(image, param):
    if param == 1:
        return cv.equalizeHist(image)
    else:
        return image


def set_gaus_rozm_jadro(image, param, sigma):
    odd = param

    if odd > 0:
        return cv.GaussianBlur(image, (odd, odd), sigma)
    else:
        return image


def set_gaus_rozm_sigma(image, param, kernel):
    odd = param

    if odd > 0:
        return cv.GaussianBlur(image, (kernel, kernel), odd)
    else:
        return image


def set_canny(image, param):
    if param > 0:
        return cv.Canny(image, param, param)
    else:
        return image


def hough(image, circles, correct_cycle):
    tp = 0
    fp = 0
    fn = 0
    print("------------------------------------------------------------------------------")
    print("------------------------Circles-----------------------------------------------")
    if circles is not None:
        image = cv.cvtColor(image, cv.CV_8UC1)
        circles = np.uint16(np.around(circles))
        print(circles)
        print(correct_cycle)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = i[2]
            # cv.circle(image, center, radius, (255, 0, 0), 3)
            square_circle = compute_square_vertices(i[0], i[1], i[2])
            cv.rectangle(image, (square_circle[0], square_circle[1]), (square_circle[2], square_circle[3]), (255, 0, 0))
            square_correct_circle = compute_square_vertices(correct_cycle[0], correct_cycle[1], correct_cycle[2])
            cv.rectangle(image, (square_correct_circle[0], square_correct_circle[1]),
                         (square_correct_circle[2], square_correct_circle[3]), (154, 250, 0))
            iou = compute_iou(square_circle, square_correct_circle)
            # iou = compute_iou(square_correct_circle, square_circle)
            print("IOU: " + str(round(iou, 2)))
            if iou >= 0.75:
                tp += 1
                cv.circle(image, center, radius, (0, 0, 255), 3)
            else:
                fp += 1
                cv.circle(image, center, radius, (255, 0, 0), 3)

        if tp == 0:
            fn = 1

        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)

        tp_str = "TP: " + str(tp) + "| "
        fp_str = "FP: " + str(fp) + "| "
        fn_str = "FN: " + str(fn) + "| "
        precision_str = "Precision: " + str(precision) + "| "
        recall_str = "Recall: " + str(recall) + "| "
        full_str1 = tp_str + fp_str + fn_str
        full_str2 = precision_str + recall_str

        if len(circles[0]) == 1:
            print("1111")
            square_circle = compute_square_vertices(circles[0][0][0], circles[0][0][1], circles[0][0][2])
            cv.rectangle(image, (square_circle[0], square_circle[1]), (square_circle[2], square_circle[3]), (255, 0, 0))
            square_correct_circle = compute_square_vertices(correct_cycle[0], correct_cycle[1], correct_cycle[2])
            cv.rectangle(image, (square_correct_circle[0], square_correct_circle[1]),
                         (square_correct_circle[2], square_correct_circle[3]), (154, 250, 0))
            iou = compute_iou(square_circle, square_correct_circle)
            iou_str = "IOU: " + str(round(iou, 2))
            full_str1 = full_str1 + iou_str

        cv.putText(image, full_str1, (1, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (8, 8, 8))  # (71, 99, 255)
        cv.putText(image, full_str2, (1, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (8, 8, 8))
        print("-----------------------------------------------")
        print(full_str1)
        print(full_str2)
        print("-----------------------------------------------")

    return image


def set_hough(image, correct_circle, houghAkumulator, houghMinVzCentKruznic, param1, param2, houghMinPolKruznic,
              houghMaxPolKruznic):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, houghAkumulator, houghMinVzCentKruznic, None, param1, param2,
                              houghMinPolKruznic, houghMaxPolKruznic)

    return hough(copy.copy(image), circles, correct_circle), circles


img_name = "duhovky/001/L/S1001L01.jpg"
img = cv.imread(cv.samples.findFile(img_name))

img = cv.imread(cv.samples.findFile(img_name))
scale_percent = 100  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

data = pd.read_csv("iris_annotation.csv")
data = data.where(data["image"] == img_name[8:]).dropna()
list = [data.columns[:, ].values.astype(str).tolist()] + data.values.tolist()
size = int(scale_percent / 100)
list = [int(list[1][1]) * size, int(list[1][2]) * size, int(list[1][3]) * size, int(list[1][4]) * size,
        int(list[1][5]) * size, int(list[1][6]) * size]

print(list)

img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

refresh_image(0, 3, 5, 0, 33, 321, 128, 322, 36, 154)
k = cv.waitKey(0)
