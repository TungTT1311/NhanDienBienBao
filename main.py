from __future__ import division
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

import time
from matplotlib import pyplot as plt
import math

MIN_MATCH_COUNT = 9
TiLeChon = 20
# Lay BIEN CAM
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienCam'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienCam = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienCam[n] = cv2.imread(join(mypath, onlyfiles[n]))
# Lay BIEN CHI DAN
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienChiDan'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienChiDan = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienChiDan[n] = cv2.imread(join(mypath, onlyfiles[n]))
# Lay BIEN NGUY HIEM
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienNguyHiem'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienNguyHiem = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienNguyHiem[n] = cv2.imread(join(mypath, onlyfiles[n]))
# Lay BIEN HIEU LENH
mypath = 'C:\\Users\\TungTT\\PycharmProjects\\untitled1\\CacLoaiBien\\BienHieuLenh'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
BienHieuLenh = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    BienHieuLenh[n] = cv2.imread(join(mypath, onlyfiles[n]))
# Mang ten bien bao:
Ten_Bien_Cam = ["Cam nguoi di bo", "Cam oto", "Toc do toi da 40"]
Ten_Bien_ChiDan = ["Chi dan: Cho quay dau", "Chi dan: Uu tien qua duong hep", "Chi dan: Duong danh chi oto"]
Ten_Bien_NguyHiem = ["Duong bi hep 2 ben", "Giao nhau voi duong khong uu tien", "Vach nui nguy hiem"]
Ten_Bien_HieuLenh = ["Chi duoc di thang", "Duong danh cho nguoi di bo", "Duong danh cho xe tho so"]
font = cv2.FONT_HERSHEY_SIMPLEX

def find_biggest_contour(image,red):
    images, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # ---truong hop tinh to nhat theo dien tich bao phu
    # # contour_size = [(cv2.contourArea(contour),contour)for contour in contours]
    # # biggest_contour = max(contour_size,key=lambda x: x[0])[1]
    #
    # ---truong hop tinh to nhat theo dien tich chiem tren frame
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
        D = w*h;
        if ((mincount <= D) & (D <= maxcount)):
            if (len(approx) > 5) & (abs(float(w/h)-1) < 0.3):
                mincount = D
                biggest_contour = contour
                if red == True:
                    shape = "Bien Cam"
                elif red == False:
                    shape = "Bien Hieu Lenh"
                # print("circle")
                # print(shape)
                # cv2.drawContours(image, biggest_contour, -1, (1, 0, 0), 2)
            elif len(approx) == 4:
                    mincount = D
                    biggest_contour = contour
                    shape = "Bien Chi Dan"
                    # cv2.drawContours(image, biggest_contour, -1, (1, 0, 0), 2)
            elif len(approx) == 3:
                    mincount = D
                    biggest_contour = contour
                    shape = "Bien Nguy Hiem"
            # print(count)
            # break

    # ----
    # shape = "undef"
    # peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # cv2.imshow("bb_ct",image)
    # if shape != "":
    return biggest_contour, shape

# ---


def SIFT_Dectect(img1, img2):
    # Initiate SIFT detector
    # orb = cv2.ORB_create(nfeatures=1500)
    sift = cv2.xfeatures2d_SIFT.create();
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
        pts1 = src_pts[mask == 1]

        percentage_similarity = len(pts1) / len(kp1) * 100
        # print(percentage_similarity)
        return percentage_similarity
        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        # print("Not matching - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return 0


# bo frame vao cho image
def DetectAndRecognize(imgFrame,red):
    Bdot = 3
    imgBlur = cv2.GaussianBlur(imgFrame, (Bdot, Bdot), 0)
    imgBlurRGB = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2RGB)
    imgBlurHSV = cv2.cvtColor(imgBlurRGB, cv2.COLOR_RGB2HSV)

    # lower_red = np.array([0, 68,134 ])
    # up_red = np.array([180, 255, 243])
    # mask = cv2.inRange(imgBlurHSV, lower_red, up_red)
    # # loai bien bao mau red
    # cv2.imshow('maskmer', mask)
    lower_red1 = np.array([0, 110, 10])
    upper_red1 = np.array([15, 250, 200])
    maskred1 = cv2.inRange(imgBlurHSV, lower_red1, upper_red1)
    lower_red2 = np.array([150, 110, 10])
    upper_red2 = np.array([250, 250, 200])
    maskred2 = cv2.inRange(imgBlurHSV, lower_red2, upper_red2)
    maskred = maskred1+maskred2

    lower_blue1 = np.array([85, 51, 99])#np.array([100, 80, 145])
    upper_blue1 = np.array([141, 255, 248])#np.array([120, 245, 255])

    maskblue = cv2.inRange(imgBlurHSV, lower_blue1, upper_blue1)
    # tim bien bao lon nhat vi no co the thay nhieu bien cung luc
    # maskref = maskred + maskblue
    cv2.imshow('maskblue', maskblue)
    cv2.imshow('maskred', maskred)

    if red == True:
        big_sign_contour, shape = find_biggest_contour(maskred, red)
    else:
        big_sign_contour, shape = find_biggest_contour(maskblue, red)

    x, y, w, h = cv2.boundingRect(big_sign_contour)

    cropped = image[y: y + h, x: x + w]
    if (shape == "Bien Chi Dan")|(shape == "Bien Cam")|(shape == "Bien Nguy Hiem")|(shape == "Bien Hieu Lenh"):
        cv2.putText(cropped, shape, (1, 30), font, 0.5, (0, 0, 255), 1)
        cv2.imshow('cropped', cropped)
    res = 0.001
    index  = 0
    i = 0
    TenBien = {}
    if shape == "Bien Chi Dan":
        # print("Bien chi dan")
        images = BienChiDan.copy()
        len_images = len(images)
        TenBien = Ten_Bien_ChiDan
    elif shape == "Bien Cam":
        # print("Bien cam")
        images = BienCam.copy()
        len_images = len(images)
        TenBien = Ten_Bien_Cam
    elif shape == "Bien Nguy Hiem":
        # print("Bien nguy hiem")
        images = BienNguyHiem.copy()
        len_images = len(images)
        TenBien = Ten_Bien_NguyHiem
    elif shape == "Bien Hieu Lenh":
        # print("Bien hieu lenh")
        images = BienHieuLenh.copy()
        len_images = len(images)
        TenBien = Ten_Bien_HieuLenh
    try:
        for i in range(len_images):
            res = SIFT_Dectect(cropped, images[i])
            if res > TiLeChon:
                index = i
                break
    except Exception as e:
        pass
    return TenBien,res, index, x, y, w, h


if __name__ == '__main__':
    # Tin hieu de phan biet circle blue vs circle red
    red = False
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = 1000 / fps
    # cap = cv2.VideoCapture("Demo.MOV")
    # delay_time = 1
    dect = True
    bTracking = False
    bLastTracking = False
    imgTracking = None
    frame_keep = 0
    # init
    index = x = y = w = h = 0
    Ti_le = 0.001

    while True:
        pre_time = time.time()
        _, image = cap.read()

        if dect:
            if bTracking == True:
                if (bLastTracking == True) :
                    # imgTracking = image[x:x + w + 1, y:y + h + 1]
                    imgMtpl = cv2.matchTemplate(imgTracking, image, cv2.TM_CCOEFF_NORMED)
                    mi_val, ma_val, mi_loc, ma_loc = cv2.minMaxLoc(imgMtpl)
                    cv2.putText(image, "%.2f" % ma_val, (1, 30), font, 3, (0, 0, 255), 3)

                    if ma_val > 0.625:
                        cv2.rectangle(image, ma_loc, (ma_loc[0] + w + 1, ma_loc[1] + h + 1), (255, 0, 0), 2)
                        frame_keep = 0

                    if (ma_val < 0.4):
                        if frame_keep > 30:
                            bLastTracking = False
                            frame_keep = 0
                        else:
                            frame_keep+=1
                else:
                    # detect
                    TenBien,Ti_le, index, x, y, w, h = DetectAndRecognize(image,red)
                    if (Ti_le > TiLeChon):
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(image, TenBien[index], (x, y), font, 0.8, (255, 255, 255), 2,
                                    cv2.LINE_AA)
                    # end of detect
                        bLastTracking = True
                        imgTracking = image[x:x + w + 1, y:y + h + 1]
            else:
                # detect
                TenBien, Ti_le, index, x, y, w, h = DetectAndRecognize(image,red)
                red = not red
                if (Ti_le > TiLeChon):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    print(TenBien[index])
                    cv2.putText(image, TenBien[index], (x , y), font, 0.8, (255, 255, 255), 2,
                                cv2.LINE_AA)
                # end of detect
                #     bLastTracking = True
                #     print("b")
                #     imgTracking = image[x:x + w + 1, y:y + h + 1]
        cv2.imshow('video', image)
        del_time = (time.time() - pre_time) * 1000
        if del_time > wait_time:
            delay_time = 1
        else:
            delay_time = wait_time - del_time
        key = cv2.waitKey(math.floor(delay_time)+1)

        if key == ord('q'):
            break
        # if key == ord('d'):
        #     dect = True
        # if key == ord('p'):
        #     dect = False
        # if key == ord('t'):
        #     bTracking = True
        # if key == ord('r'):
        #     bTracking == False

    cv2.destroyAllWindows()
