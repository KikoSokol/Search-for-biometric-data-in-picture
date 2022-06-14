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


# def compute_info(image, circles, x, y, radius):


def refresh_image():
    image = copy.copy(img)

    hist = cv.getTrackbarPos('histogram', 'set')
    image = set_histogram(image, hist)

    gaus_kerner = cv.getTrackbarPos('grj', 'set')
    image = set_gaus_rozm_jadro(image, gaus_kerner, 1)

    gaus_sigma = cv.getTrackbarPos('grs', 'set')
    image = set_gaus_rozm_sigma(image, gaus_sigma, 2)

    canny = cv.getTrackbarPos('canny', 'set')
    image = set_canny(image, canny)

    is_houg = cv.getTrackbarPos('setHoug', 'set')

    if is_houg == 1:
        result = set_hough(image, list[0:3])
        image = result[0]

    image = add_correct_cicle(image, list, 1)
    cv.imshow(img_name, image)


def histogram(a):
    refresh_image()


def get_odd_number(a):
    if a == 0:
        return 0
    if a % 2 == 0:
        return a + 1

    return a


def gausRozmJadro(a):
    refresh_image()


def gausRozmSigma(a):
    refresh_image()


def canny(a):
    refresh_image()


def hough(image, circles, correct_cycle):
    tp = 0
    fp = 0
    fn = 0
    # print("------------------------------------------------------------------------------")
    # print("------------------------Circles-----------------------------------------------")
    if circles is not None:
        image = cv.cvtColor(image, cv.CV_8UC1)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = i[2]
            # cv.circle(image, center, radius, (255, 0, 0), 3)
            square_circle = compute_square_vertices(i[0].item(), i[1].item(), i[2].item())
            cv.rectangle(image, (square_circle[0], square_circle[1]), (square_circle[2], square_circle[3]), (255, 0, 0))
            square_correct_circle = compute_square_vertices(correct_cycle[0], correct_cycle[1], correct_cycle[2])
            cv.rectangle(image, (square_correct_circle[0], square_correct_circle[1]),
                         (square_correct_circle[2], square_correct_circle[3]), (154, 250, 0))
            # iou = compute_iou(square_circle, square_correct_circle)
            iou = compute_iou(square_correct_circle, square_circle)
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
            # print("1111")
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


def houghAkumulator(a):
    refresh_image()


def houghMinVzCentKruznic(a):
    refresh_image()


def param1(a):
    refresh_image()


def param2(a):
    refresh_image()


def houghMinPolKruznic(a):
    refresh_image()


def houghMaxPolKruznic(a):
    refresh_image()


def setHoug(a):
    refresh_image()


def set_histogram(image, param):
    if param == 1:
        return cv.equalizeHist(image)
    else:
        return image


def set_gaus_rozm_jadro(image, param, window):
    odd = get_odd_number(param)
    if param != odd and window == 1:
        cv.setTrackbarPos('grj', 'set', odd)
    elif param != odd and window == 2:
        cv.setTrackbarPos('grj_2', 'set2', odd)

    if odd > 0:
        return cv.GaussianBlur(image, (odd, odd), get_odd_number(cv.getTrackbarPos('grs', 'set')))
    else:
        return image


def set_gaus_rozm_sigma(image, param, window):
    odd = get_odd_number(param)
    if param != odd and window == 1:
        cv.setTrackbarPos('grs', 'set', odd)
    elif param != odd and window == 2:
        cv.setTrackbarPos('grs_2', 'set2', odd)

    if odd > 0:
        kernel = get_odd_number(cv.getTrackbarPos('grj', 'set'))
        return cv.GaussianBlur(image, (kernel, kernel), odd)
    else:
        return image


def set_canny(image, param):
    if param > 0:
        return cv.Canny(image, param, param)
    else:
        return image


def set_hough(image, correct_circle):
    houghAkumulator = cv.getTrackbarPos('houghAkumulator', 'set')
    houghMinVzCentKruznic = cv.getTrackbarPos('houghMinVzCentKruznic', 'set')
    houghMinPolKruznic = cv.getTrackbarPos('houghMinPolKruznic', 'set')
    param1 = cv.getTrackbarPos('param1', 'set')
    param2 = cv.getTrackbarPos('param2', 'set')
    houghMaxPolKruznic = cv.getTrackbarPos('houghMaxPolKruznic', 'set')

    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, houghAkumulator, houghMinVzCentKruznic, None, param1, param2,
                              houghMinPolKruznic, houghMaxPolKruznic)

    return hough(copy.copy(image), circles, correct_circle), circles


img_name = "duhovky/001/L/S1001L01.jpg"
# img_name = "duhovky/001/R/S1001R01.jpg"
# img_name = "duhovky/008/L/S1008L01.jpg"
# img_name = "duhovky/011/L/S1011L01.jpg"

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

# resize image
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
if img is None:
    sys.exit("Could not read the image.")

set_window = cv.namedWindow('set', cv.WINDOW_NORMAL)

cv.createTrackbar('histogram', 'set', 0, 1, histogram)
cv.createTrackbar('grj', 'set', 0, 255, gausRozmJadro)
cv.createTrackbar('grs', 'set', 0, 255, gausRozmSigma)
cv.createTrackbar('canny', 'set', 0, 1000, canny)
cv.createTrackbar('setHoug', 'set', 0, 1, setHoug)
cv.createTrackbar('houghAkumulator', 'set', 1, 50, houghAkumulator)
cv.createTrackbar('houghMinVzCentKruznic', 'set', 1, 500, houghMinVzCentKruznic)
cv.createTrackbar('param1', 'set', 1, 500, param1)
cv.createTrackbar('param2', 'set', 1, 500, param2)
cv.createTrackbar('houghMinPolKruznic', 'set', 1, 500, houghMinPolKruznic)
cv.createTrackbar('houghMaxPolKruznic', 'set', 1, 500, houghMaxPolKruznic)
cv.imshow(img_name, img)


# k = cv.waitKey(0)

##########################################################################################################
def refresh_image_2():
    image = copy.copy(img2)

    hist = cv.getTrackbarPos('histogram_2', 'set2')
    image = set_histogram(image, hist)

    gaus_kerner = cv.getTrackbarPos('grj_2', 'set2')
    image = set_gaus_rozm_jadro(image, gaus_kerner, 2)

    gaus_sigma = cv.getTrackbarPos('grs_2', 'set2')
    image = set_gaus_rozm_sigma(image, gaus_sigma, 2)

    canny = cv.getTrackbarPos('canny_2', 'set2')
    image = set_canny(image, canny)

    is_houg = cv.getTrackbarPos('setHoug_2', 'set2')

    if is_houg == 1:
        image = set_hough_2(image, list[3:6])

    image = add_correct_cicle(image, list, 2)
    cv.imshow(img_name + " - 2", image)


def histogram_2(a):
    refresh_image_2()


def get_odd_number_2(a):
    if a == 0:
        return 0
    if a % 2 == 0:
        return a + 1

    return a


def gausRozmJadro_2(a):
    refresh_image_2()


def gausRozmSigma_2(a):
    refresh_image_2()


def canny_2(a):
    refresh_image_2()


def hough_2(image, circles):
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


def houghAkumulator_2(a):
    refresh_image_2()


def houghMinVzCentKruznic_2(a):
    refresh_image_2()


def param1_2(a):
    refresh_image_2()


def param2_2(a):
    refresh_image_2()


def houghMinPolKruznic_2(a):
    refresh_image_2()


def houghMaxPolKruznic_2(a):
    refresh_image_2()


def setHoug_2(a):
    refresh_image_2()


def set_hough_2(image, correct_circle):
    houghAkumulator = cv.getTrackbarPos('houghAkumulator_2', 'set2')
    houghMinVzCentKruznic = cv.getTrackbarPos('houghMinVzCentKruznic_2', 'set2')
    houghMinPolKruznic = cv.getTrackbarPos('houghMinPolKruznic_2', 'set2')
    param1 = cv.getTrackbarPos('param1_2', 'set2')
    param2 = cv.getTrackbarPos('param2_2', 'set2')
    houghMaxPolKruznic = cv.getTrackbarPos('houghMaxPolKruznic_2', 'set2')

    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, houghAkumulator, houghMinVzCentKruznic, None, param1, param2,
                              houghMinPolKruznic, houghMaxPolKruznic)

    return hough(copy.copy(image), circles, correct_circle)


img2 = cv.imread(cv.samples.findFile(img_name))
scale_percent2 = scale_percent  # 100  # percent of original size
width2 = int(img2.shape[1] * scale_percent2 / 100)
height2 = int(img2.shape[0] * scale_percent2 / 100)
dim2 = (width2, height2)

# resize image
img2 = cv.resize(img2, dim2, interpolation=cv.INTER_AREA)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
if img2 is None:
    sys.exit("Could not read the image.")

set_window2 = cv.namedWindow('set2', cv.WINDOW_NORMAL)

cv.createTrackbar('histogram_2', 'set2', 0, 1, histogram_2)
cv.createTrackbar('grj_2', 'set2', 0, 255, gausRozmJadro_2)
cv.createTrackbar('grs_2', 'set2', 0, 255, gausRozmSigma_2)
cv.createTrackbar('canny_2', 'set2', 0, 1000, canny_2)
cv.createTrackbar('setHoug_2', 'set2', 0, 1, setHoug_2)
cv.createTrackbar('houghAkumulator_2', 'set2', 1, 50, houghAkumulator_2)
cv.createTrackbar('houghMinVzCentKruznic_2', 'set2', 1, 500, houghMinVzCentKruznic_2)
cv.createTrackbar('param1_2', 'set2', 1, 500, param1_2)
cv.createTrackbar('param2_2', 'set2', 1, 500, param2_2)
cv.createTrackbar('houghMinPolKruznic_2', 'set2', 1, 500, houghMinPolKruznic_2)
cv.createTrackbar('houghMaxPolKruznic_2', 'set2', 1, 500, houghMaxPolKruznic_2)
cv.imshow(img_name + " - 2", img)
k2 = cv.waitKey(0)

# add_correct_cicle(None, img_name)
