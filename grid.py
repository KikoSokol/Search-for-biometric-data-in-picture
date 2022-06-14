import cv2 as cv
import sys
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm


def write_head(file):
    file.write("obrazok, kernel, sigma, akumulator, cent, param1, param2, min_pol, max_pol, precision, recall\n")


def write_to_file(file, img_name, kernel, sigma, akumulator, cent, param1, param2, min_pol, max_pol, precision, recall):
    c = ", "

    file.write(str(img_name) + c + str(kernel) + c + str(sigma) + c + str(akumulator) + c + str(cent) + c + str(param1)
               + c + str(param2) + c + str(min_pol) + c + str(max_pol) + c + str(precision) + c + str(recall))
    file.write("\n")


def compute_square_vertices(x, y, radius):
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


def set_gaus_rozm_jadro(image, param_kernel, param_sigma):
    return cv.GaussianBlur(image, (param_kernel, param_kernel), param_sigma)


def set_gaus_rozm_sigma(image, param_kernel, param_sigma):
    return cv.GaussianBlur(image, (param_kernel, param_kernel), param_sigma)


def set_hough(image, akumulator, vz_cent, param1, param2, min_pol, max_pol):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, akumulator, vz_cent, None, param1, param2,
                              min_pol, max_pol)

    return circles


def hough(circles, correct_cycle):
    tp = 0
    fp = 0
    fn = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            square_circle = compute_square_vertices(i[0], i[1], i[2])
            square_correct_circle = compute_square_vertices(correct_cycle[0], correct_cycle[1], correct_cycle[2])
            # iou = compute_iou(square_circle, square_correct_circle)
            iou = compute_iou(square_correct_circle, square_circle)
            # print("IOU: " + str(round(iou, 2)))
            if iou >= 0.75:
                tp += 1
            else:
                fp += 1

        if tp == 0:
            fn = 1

        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)
        return (tp, fp, fn), precision, recall

    return (0, 0, 0), 0, 0


img_name = "duhovky/001/L/S1001L01.jpg"

img = cv.imread(cv.samples.findFile(img_name))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

data = pd.read_csv("iris_annotation.csv")
data = data.where(data["image"] == img_name[8:]).dropna()
list = [data.columns[:, ].values.astype(str).tolist()] + data.values.tolist()
list = [int(list[1][1]), int(list[1][2]), int(list[1][3]), int(list[1][4]),
        int(list[1][5]), int(list[1][6])]

zrnicka = open("zrnicka.txt", "w")
duhovka = open("duhovka.txt", "w")

write_head(zrnicka)
write_head(duhovka)

for kernel in tqdm(range(4)):
    print("kernel " + str(kernel))
    k = 0
    if kernel == 0:
        k = 1
    elif kernel == 1:
        k = 3
    elif kernel == 2:
        k = 5
    elif kernel == 3:
        k = 9

    for sigma in tqdm(range(11)):
        print("sigma " + str(sigma))
        image_sigma = set_gaus_rozm_sigma(img, k, sigma)

        for akumulator in range(1, 20):
            print("akumulator " + str(akumulator))
            for cent in range(1, 20):
                print("cent " + str(cent))
                for param1 in range(1, 20):
                    print("param1 " + str(param1))
                    for param2 in range(1, 20):
                        print("param2 " + str(param2))
                        for min_pol in range(1, 20):
                            print("min_pol " + str(min_pol))
                            for max_pol in range(1, 20):
                                print("max_pol " + str(max_pol))
                                circles = set_hough(image_sigma, akumulator, cent, param1, param2, min_pol, max_pol)

                                result_zrnicka = hough(circles, list[0: 3])
                                result_duhovka = hough(circles, list[3:6])

                                if result_zrnicka[1] == 1:
                                    write_to_file(zrnicka, img_name, k, sigma, akumulator, cent, param1, param2,
                                                  min_pol, max_pol, result_zrnicka[1], result_zrnicka[2])

                                if result_duhovka[1] == 1:
                                    write_to_file(duhovka, img_name, k, sigma, akumulator, cent, param1, param2,
                                                  min_pol, max_pol, result_duhovka[1], result_duhovka[2])

zrnicka.close()
duhovka.close()
