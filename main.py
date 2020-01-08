from __future__ import division
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import math

MIN_MATCH_COUNT = 5
TiLeChon = 10
font = cv2.FONT_HERSHEY_SIMPLEX
# Get BIEN CAM
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienCam'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienCam = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienCam[n] = cv2.imread(join(mypath, onlyfiles[n]))

# Get BIEN CHI DAN
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienChiDan'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienChiDan = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienChiDan[n] = cv2.imread(join(mypath, onlyfiles[n]))

# Get BIEN NGUY HIEM
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienNguyHiem'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienNguyHiem = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienNguyHiem[n] = cv2.imread(join(mypath, onlyfiles[n]))

# Get BIEN HIEU LENH
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienHieuLenh'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienHieuLenh = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienHieuLenh[n] = cv2.imread(join(mypath, onlyfiles[n]))

# Array saved traffic sign name:
Ten_Bien_Cam = ["Cam nguoi di bo", "Cam oto", "Toc do toi da 40"]
Ten_Bien_ChiDan = ["Cho quay dau", "Uu tien qua duong hep", "Duong danh cho oto"]
Ten_Bien_NguyHiem = ["Duong bi hep 2 ben", "Giao nhau voi duong khong uu tien", "Vach nui nguy hiem"]
Ten_Bien_HieuLenh = ["Chi duoc di thang", "Duong danh cho nguoi di bo", "Noi giao nhau theo vong xuyen"]



def find_biggest_contour(image,red):
    images, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mincount = 8060
    maxcount = 100264
    biggest_contour = None
    shape = ""
    pi = 3.14
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        D = w*h
        if ((mincount <= D) & (D <= maxcount)):
            # Circle
            if (len(approx) > 5) & (abs(float(w/h)-1) < 0.3):
                mincount = D
                biggest_contour = contour
                if red == True:
                    shape = "Bien Cam"
                elif red == False:
                    shape = "Bien Hieu Lenh"
            # Rectangle
            elif len(approx) == 4:
                    mincount = D
                    biggest_contour = contour
                    shape = "Bien Chi Dan"
            # Triangle
            elif len(approx) == 3:
                    mincount = D
                    biggest_contour = contour
                    shape = "Bien Nguy Hiem"

    return biggest_contour, shape

# ---

def SIFT_Dectect(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d_SIFT.create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # Count the inlier points
        pts1 = src_pts[mask == 1]
        # Same ratio
        percentage_similarity = len(pts1) / len(kp1) * 100
        return percentage_similarity
    else:
        return 0
# ---
def DetectAndRecognize(imgFrame, red):
    # Color filter
    max = 0
    Bdot = 3
    imgBlur = cv2.GaussianBlur(imgFrame, (Bdot, Bdot), 0)
    imgBlurRGB = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2RGB)
    imgBlurHSV = cv2.cvtColor(imgBlurRGB, cv2.COLOR_RGB2HSV)
    # Red filter
    lower_red1 = np.array([0, 110, 10])
    upper_red1 = np.array([15, 250, 200])
    maskred1 = cv2.inRange(imgBlurHSV, lower_red1, upper_red1)
    lower_red2 = np.array([150, 110, 10])
    upper_red2 = np.array([250, 250, 200])
    maskred2 = cv2.inRange(imgBlurHSV, lower_red2, upper_red2)
    maskred = maskred1 + maskred2
    # Blue filter
    lower_blue1 = np.array([100, 80, 145])
    upper_blue1 = np.array([120, 245, 255])
    maskblue = cv2.inRange(imgBlurHSV, lower_blue1, upper_blue1)

    cv2.imshow('maskblue', maskblue)
    cv2.imshow('maskred', maskred)
    # Handling rotation maskred and markblue
    if red == True:
        big_sign_contour, shape = find_biggest_contour(maskred, red)
    else:
        big_sign_contour, shape = find_biggest_contour(maskblue, red)

    x, y, w, h = cv2.boundingRect(big_sign_contour)
    # Traffic sign photo taken from video
    cropped = image[y: y + h, x: x + w]
    # Get photo albums of that type of traffic sign
    res = 0.001
    index  = 0
    i = 0
    TenBien = {}
    if shape == "Bien Chi Dan":
        images = BienChiDan.copy()
        len_images = len(images)
        TenBien = Ten_Bien_ChiDan
    elif shape == "Bien Cam":
        images = BienCam.copy()
        len_images = len(images)
        TenBien = Ten_Bien_Cam
    elif shape == "Bien Nguy Hiem":
        images = BienNguyHiem.copy()
        len_images = len(images)
        TenBien = Ten_Bien_NguyHiem
    elif shape == "Bien Hieu Lenh":
        images = BienHieuLenh.copy()
        len_images = len(images)
        TenBien = Ten_Bien_HieuLenh
    # Compare using SIFT feature
    try:
        for i in range(len_images):
            res = SIFT_Dectect(cropped, images[i])
            if i == len_images:
                break
            if res > max:
                max = res
                index = i
    except Exception as e:
        pass
    return TenBien,max, index, x, y, w, h, shape


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = 1000 / fps
    # Init
    red = False  # Signal to distinguish circle blue vs circle red
    frame_keep = 0
    index = x = y = w = h = 0
    Ti_le = 0.001
    time_everage = 0
    while True:
        pre_time = time.time()
        _, image = cap.read()

        TenBien, Ti_le, index, x, y, w, h, Loai = DetectAndRecognize(image,red)

        red = not red
        # Draw frames, print the name of traffic signs
        if (Ti_le > TiLeChon):   # Avoid false detection
            cropped_1 = image[y: y + h, x: x + w + 70]
            In = Loai + ": " + TenBien[index]
            cv2.putText(cropped_1, In, (0, 20), font, 0.5, (0, 0, 0), 1)
            cv2.imshow('TenBienBao', cropped_1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('video', image)
        del_time = (time.time() - pre_time) * 1000
        if del_time > wait_time:
            delay_time = 1
        else:
            delay_time = wait_time - del_time
        key = cv2.waitKey(math.floor(delay_time)+1)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
