import wsgiref.validate

import cv2 as cv
import sys
import copy
import numpy as np
from time import sleep
from tqdm import tqdm


def set_histogram(image, param):
    if param == 1:
        return cv.equalizeHist(image)
    else:
        return image


def set_gaus_rozm_jadro(image, param_kernel, param_sigma):
    return cv.GaussianBlur(image, (param_kernel, param_kernel), param_sigma)


def set_gaus_rozm_sigma(image, param_kernel, param_sigma):
    return cv.GaussianBlur(image, (param_kernel, param_kernel), param_sigma)


def set_canny(image, param):
    return cv.Canny(image, param, param)


def hough(image, circles):
    if circles is not None:
        image = cv.cvtColor(image, cv.CV_8UC1)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 0), 3)

    return image


def set_hough(image, akumulator, vz_cent, param1, param2, min_pol, max_pol):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, akumulator, vz_cent, None, param1, param2,
                              min_pol, max_pol)

    return hough(copy.copy(image), circles)


def write_to_file(file, id, hist, grj, grs, canny, akum, vz_cent, param1, param2, min_pol, max_pol):
    file.write("ID: ")
    file.write(str(id))
    file.write('\n')

    file.write("Histogram: " + str(hist))
    file.write('\n')

    file.write("Gausovo rozmazanie jadro: " + str(grj))
    file.write('\n')

    file.write("Gausovo rozmazanie sigma: " + str(grs))
    file.write('\n')

    file.write("canny: " + str(canny))
    file.write('\n')

    file.write("Akumulator: " + str(akum))
    file.write('\n')

    file.write("Min vz cent kruznic: " + str(vz_cent))
    file.write('\n')

    file.write("Param1: " + str(param1))
    file.write('\n')

    file.write("Param2: " + str(param2))
    file.write('\n')

    file.write("Min polomer: " + str(min_pol))
    file.write('\n')

    file.write("Max polomer: " + str(max_pol))
    file.write('\n')

    file.write("---------------------------------------------------------------------------------------------------")
    file.write('\n')
    file.write("---------------------------------------------------------------------------------------------------")
    file.write('\n')


img_name = "duhovky/008/L/S1008L01.jpg"

img = cv.imread(cv.samples.findFile(img_name))

scale_percent = 200  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cv.imshow(img_name, img)
k = cv.waitKey(0)

id = 0

file = open("combination.txt", "w")

for h in tqdm(range(2)):
    hist = set_histogram(img, h)
    print(h)
    for gj in tqdm(range(0, 255 * 2)):

        if gj == 0:
            gausJadro = hist
        else:
            if gj % 2 == 0:
                continue
            else:
                gausJadro = set_gaus_rozm_jadro(hist, gj, 1)

        for gs in tqdm(range(255 * 2)):

            if gs == 0:
                gausSigma = gausJadro
            else:
                if gs % 2 == 0:
                    continue
                else:
                    gausSigma = set_gaus_rozm_sigma(gausJadro, gj, gs)

            for cann in tqdm(range(255)):
                canny = set_canny(gausSigma, cann)

                for akum in tqdm(range(1, 1001)):

                    for vz_cent in tqdm(range(1, 1001)):

                        for param1 in tqdm(range(1, 1001)):

                            for param2 in tqdm(range(1, 1001)):

                                for min_pol in tqdm(range(1, 1001)):

                                    for max_pol in tqdm(range(1, 1001)):
                                        id += 1
                                        write_to_file(file, id, h, gj, gs, cann, akum, vz_cent, param1, param2, min_pol,
                                                      max_pol)

file.close()
