import numpy as np
import cv2
import os
from os import listdir
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import resource
from openfacetests import cropFace, cnnDetect
import random
from subprocess import call
import openface
import dlib
from skimage.util import random_noise
import itertools

projectPath = "/Users/sidharthmenon/Desktop/Summer 2018/open-door/liveness-dataset/"


# target = projectPath + "7/Down/Vl_NT_HS_wg_E_7_120.mp4"
# target2 = projectPath + "7/Left/G_NT_HS_wg_E_7_6.mp4"
# target3 = projectPath + "7/Right/G_NT_5s_wg_E_7_1.mp4"



# converts takes image, corrects for direction
def toVertical(img, dir):
    if dir == "Down":
        return cv2.flip(img, -1)
    elif dir == "Up":
        return img
    else:
        img = cv2.transpose(img)
        if dir == "Right":
            return cv2.flip(img, 0)
        else:
            return cv2.flip(img, 1)

# for i in range(1, len(folder_array)):
#     print "jello"

# for index in map(str, [7, 9, 10, 14, 15, 18, 20, 23]):
#     for dir in ["Up", "Down", "Left", "Right"]:
#         folderPath = projectPath + index + "/" + dir
#         for fileName in os.listdir(folderPath):
#             filePath = folderPath + "/" + fileName
# vid = cv2.VideoCapture(target3)
# res = []
# frameNum = 0
# while(frameNum < 1):
#     _, frame = vid.read()
#     frame = toImg(frame, "Right")
#     cv2.imshow("Vertical?", frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     frame = cv2.resize(frame, (96, 96))
#     frameNum += 1
# vid.release()
# cv2.destroyAllWindows()

# vid = cv2.VideoCapture(target)
# x = 0
# print vid.get(cv2.cv.CV_CAP_PROP_FPS)
# while(x < 1):
#     _, frame = vid.read()
#     frame = toVertical(frame, "Down")
#     frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
#     big = cv2.resize(frame, (0,0), fx=1.8, fy=1.8)
#     vid.release()
#     cv2.destroyAllWindows()
#     x = x + 1

_, hdata_lim = resource.getrlimit(resource.RLIMIT_DATA)
resource.setrlimit(resource.RLIMIT_DATA, (hdata_lim, hdata_lim))
_, hfile_lim = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (hfile_lim, hfile_lim))
def prepData():
    X_data = []
    Y_data = []
    yEval = lambda s: 1 if (s[0] == "G") else 0
    for index in map(str, [2, 3, 4, 5, 6, 11, 12, 13, 16, 17, 21, 22, 7, 9, 10, 14, 15, 18, 20, 23]):
        for dir in ["Up", "Down", "Left", "Right"]:
            folderPath = projectPath + index + "/" + dir
            for fileName in os.listdir(folderPath):
                filePath = folderPath + "/" + fileName
                vid = cv2.VideoCapture(filePath)
                res = []
                frameNum = 0
                while(frameNum < 15):
                    frameNum = frameNum + 1
                    retVal, frame = vid.read()
                    if not retVal:
                        print (index, dir, fileName, frameNum)
                    frame = toVertical(frame, dir)
                    frame = cv2.resize(frame, None, fx=1.8, fy=1.8)
                    frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                    res.append(frame)
                X_data.append(np.asarray(res))
                Y_data.append(np.asarray([yEval(fileName)]))
                vid.release()
                cv2.destroyAllWindows()
                print "+"
            print "++"
        print "+++"
    X_data = np.asarray(X_data)
    Y_data = np.asarray(Y_data)
    print X_data.shape
    print Y_data.shape
    return X_data, Y_data

def skip(vid, num):
    if num > 0:
        vid.read()
        return skip(vid, num - 1)


def checkBlackFrames():
        first = []
        total_pixels = 96*96
        denom = 96*96.
        percentBlack = lambda img: round((total_pixels-cv2.countNonZero(img))/denom, 2)
        isBlack = lambda img: True if percentBlack(img) > .90 else False
        for index in map(str, [2, 3, 4, 5, 6, 11, 12, 13, 16, 17,
            21, 22, 7, 9, 10, 14, 15, 18, 20, 23]):
            for dir in ["Up", "Down", "Left", "Right"]:
                folderPath = projectPath + index + "/" + dir
                for fileName in random.sample(os.listdir(folderPath), 3):
                    filePath = folderPath + "/" + fileName
                    vid = cv2.VideoCapture(filePath)
                    frameNum = 0
                    blackList = []
                    while(frameNum < 40):
                        retVal, frame = vid.read()
                        if not retVal:
                            print (index, dir, fileName, frameNum)
                        else:
                            r, _, _ = cv2.split(frame)
                            r = cv2.resize(r, (96, 96), interpolation=cv2.INTER_AREA)
                            if isBlack(r):
                                blackList.append(frameNum)
                            # if not(hitFirstNonBlack):
                            #     if not(isBlack(r)):
                            #         first.append(frameNum)
                            #         hitFirstNonBlack = True
                            # elif not(hitLastNonBlack):
                            #     if isBlack(r):
                            #         last.append(frameNum - 1)
                            #         hitLastNonBlack = True
                            #     elif frameNum == 39:
                            #         last.append(frameNum)
                        frameNum += 1
                    vid.release()
                    cv2.destroyAllWindows()
                    if len(blackList) > 0:
                        first.append((index, dir, fileName, blackList))
                    print "+"
                print "++"
            print "+++"
        return first


# prep data
def prepSeqData(skipFrame=5):
        X_data = []
        Y_data = []
        NoFaceData = []
        NoFaceData_Labels = []
        record = []
        total_pixels = 96*96
        denom = 96*96.
        percentBlack = lambda img: round((total_pixels-cv2.countNonZero(img))/denom, 2)
        split = lambda img: cv2.split(img)[0]
        isBlack = lambda img: True if percentBlack(split(img)) > .90 else False
        yEval = lambda s: 1 if (s[0] == "G") else 0
        for index in map(str, [2, 3, 4, 5, 6, 11, 12, 13, 16, 17,
            21, 22, 7, 9, 10, 14, 15, 18, 20, 23]):
            for dir in ["Up", "Down", "Left", "Right"]:
                folderPath = projectPath + index + "/" + dir
                for fileName in os.listdir(folderPath):
                    filePath = folderPath + "/" + fileName
                    cap = cv2.VideoCapture(filePath)
                    frameNum = 0
                    tempRec = []
                    while(frameNum < 10):
                        frameNum = frameNum + 1
                        retVal, frame = cap.read()
                        if not retVal:
                            print (index, dir, fileName, frameNum)
                        else:
                            if isBlack(frame):
                                print "Before Vert {0}".format((index, dir, fileName, frameNum))
                            frame = toVertical(frame, dir)
                            if isBlack(frame):
                                print "After Vert {0}".format((index, dir, fileName, frameNum))
                            frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                            if isBlack(frame):
                                print "Resize {0}".format((index, dir, fileName, frameNum))
                            face = cropFace(frame)
                            yLabel = np.asarray([yEval(fileName)])
                            if not (face is None):
                                if isBlack(face):
                                    print "Face {0}".format((index, dir, fileName, frameNum))
                                face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
                                X_data.append(face)
                                Y_data.append(yLabel)
                            else:
                                NoFaceData.append(frame)
                                NoFaceData_Labels.append(yLabel)
                    if len(tempRec) > 0:
                        record.append((index, dir, fileName, tempRec))
                    cap.release()
                    cv2.destroyAllWindows()
        a, b, c, d = map(np.asarray, (X_data, Y_data, NoFaceData, NoFaceData_Labels))
        return a, b, c, d, record

def storeSeqData():
    data = prepSeqData()
    np.save("data.npy", data)
    print "saved"


def readRecord(rec, init=5):
    max = init
    total_pixels = 96*96
    denom = 96*96.
    percentBlack = lambda img: round((total_pixels-cv2.countNonZero(img))/denom, 2)
    isBlack = lambda img: True if percentBlack(img) > .90 else False
    split = lambda img: cv2.split(img)[0]
    for (ind, dir, fileName, frameNums) in rec:
        path = projectPath + ind + "/" + dir + "/" + fileName
        cap = cv2.VideoCapture(path)
        frameNum = 1
        notHit = True
        skip(cap, init)
        while notHit:
            retVal, frame = cap.read()
            if not retVal:
                print (ind, dir, fileName, frameNum + 5)
                notHit = False
            else:
                if not(isBlack(split(frame))):
                    notHit = False
                    if (5 + frameNum) > max:
                        max = 5 + frameNum
            frameNum += 1
        print "+"
    return max

# TODO: align, normalize (stuff in process_image below)...probably should only do that if
# a face is detected? or should you align the image regardless? idk. You make the call
# prep data template for transfer learning models
def prepData_Specific(nickName, first_resize=None, histEqualize=True, faceDetect=True,
                      scale=1.0, align=False):
        X_data = []
        Y_data = []
        NoFaceData = []
        NoFaceData_Labels = []
        total_pixels = 96*96
        denom = 96*96.
        percentBlack = lambda img: round((total_pixels-cv2.countNonZero(img))/denom, 2)
        split = lambda img: cv2.split(img)[0]
        isBlack = lambda img: True if percentBlack(split(img)) > .90 else False
        yEval = lambda s: 1 if (s[0] == "G") else 0
        for index in map(str, [2, 3, 4, 5, 6, 11, 12, 13, 16, 17,
            21, 22, 7, 9, 10, 14, 15, 18, 20, 23]):
            for dir in ["Up", "Down", "Left", "Right"]:
                folderPath = projectPath + index + "/" + dir
                for fileName in os.listdir(folderPath):
                    filePath = folderPath + "/" + fileName
                    cap = cv2.VideoCapture(filePath)
                    frameNum = 0
                    while(frameNum < 10):
                        frameNum = frameNum + 1
                        retVal, frame = cap.read()
                        if not retVal:
                            if frameNum == 1:
                                print ("retVal bad", index, dir, fileName, frameNum)
                        elif isBlack(frame):
                                frameNum = frameNum - 1
                        else:
                            frame = toVertical(frame, dir)
                            if not (first_resize is None):
                                frame = cv2.resize(frame, first_resize, interpolation=cv2.INTER_CUBIC)
                            face = cnnDetect(frame, equalize_hist=histEqualize, faceDetect=faceDetect, scale_factor=scale)
                            yLabel = yEval(fileName)
                            if not (face is None):
                                face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
                                if align:
                                    bb = dlib.rectangle(0, 0, 96, 96)
                                    face = prepData_Specific.align.align(96, face, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                                    face = np.around(face/255.0, decimals=12)
                                X_data.append(face)
                                Y_data.append(yLabel)
                            else:
                                frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_CUBIC)
                                NoFaceData.append(frame)
                                NoFaceData_Labels.append(yLabel)
                    cap.release()
                    cv2.destroyAllWindows()
        fileName = "{0}.npz".format(nickName)
        call(["touch", fileName])
        return map(np.asarray, (X_data, Y_data, NoFaceData, NoFaceData_Labels))

facePredictor = '/Users/sidharthmenon/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
prepData_Specific.align = openface.AlignDlib(facePredictor)

def prepData_Specific_pt1(nickName, first_resize=None):
        X_data = []
        Y_data = []
        NoFaceData = []
        NoFaceData_Labels = []
        total_pixels = 96*96
        denom = 96*96.
        percentBlack = lambda img: round((total_pixels-cv2.countNonZero(img))/denom, 2)
        split = lambda img: cv2.split(img)[0]
        isBlack = lambda img: True if percentBlack(split(img)) > .90 else False
        yEval = lambda s: 1 if (s[0] == "G") else 0
        for index in map(str, [2, 3, 4, 5, 6, 11, 12, 13, 16, 17,
            21, 22, 7, 9, 10, 14, 15, 18, 20, 23]):
            for dir in ["Up", "Down", "Left", "Right"]:
                folderPath = projectPath + index + "/" + dir
                for fileName in os.listdir(folderPath):
                    filePath = folderPath + "/" + fileName
                    cap = cv2.VideoCapture(filePath)
                    frameNum = 0
                    while(frameNum < 10):
                        frameNum = frameNum + 1
                        retVal, frame = cap.read()
                        if not retVal:
                            if frameNum == 1:
                                print ("retVal bad", index, dir, fileName, frameNum)
                        elif isBlack(frame):
                                frameNum = frameNum - 1
                        else:
                            frame = toVertical(frame, dir)
                            if not (first_resize is None):
                                frame = cv2.resize(frame, first_resize, interpolation=cv2.INTER_CUBIC)
                            yLabel = yEval(fileName)
                            X_data.append(frame)
                            Y_data.append(yLabel)
                    cv2.destroyAllWindows()
                    cap.release()
        fileName = "{0}.npz".format(nickName)
        call(["touch", fileName])
        return map(np.asarray, (X_data, Y_data))

def prepData_Specific_pt2(fileName=None, rx=None, ry=None, histEqualize=True, faceDetect=True, scale=1.0, align=False):
    rawImgs = []
    rawImg_Labels = []
    assert ((fileName is None) != ((rx is None) and (ry is None)))
    if fileName is None:
        rawImgs = rx
        rawImg_Labels = ry
    else:
        npz_file = np.load(fileName)
        rawImgs = npz_file['rx']
        rawImg_Labels = npz_file['ry']
    X_data = []
    Y_data = []
    NoFaceData = []
    NoFaceData_Labels = []
    for i in range(rawImgs.shape[0]):
        frame = rawImgs[i, ::]
        face = cnnDetect(frame, equalize_hist=histEqualize, faceDetect=faceDetect, scale_factor=scale)
        yLabel = rawImg_Labels[i]
        if not (face is None):
            face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
            if align:
                bb = dlib.rectangle(0, 0, 96, 96)
                face = prepData_Specific_pt2.align.align(96, face, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                face = np.around(face/255.0, decimals=12)
            X_data.append(face)
            Y_data.append(yLabel)
        else:
            frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_CUBIC)
            NoFaceData.append(frame)
            NoFaceData_Labels.append(yLabel)
    return map(np.asarray, (X_data, Y_data, NoFaceData, NoFaceData_Labels))

facePredictor = '/Users/sidharthmenon/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
prepData_Specific_pt2.align = openface.AlignDlib(facePredictor)

def addNoise(img, numTimes):
    tot = []
    for i in xrange(numTimes):
        tot.append(random_noise(img, mode='s&p', salt_vs_pepper=0.2))
    return tot


def augmentData(X_data, Y_data, n):
    add_imgs = []
    add_labels = []
    yEval = lambda s: 1 if 'fake' in s else 0
    for fileName in os.listdir('./image-test'):
        if not ('Store' in fileName):
            yLabel = yEval(fileName)
            img = cv2.imread('./image-test/{0}'.format(fileName), 1)
            if yLabel == 1 or 'IMG' in fileName:
                img = toVertical(img, 'Left')
            img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
            face = cnnDetect(img, equalize_hist=True, faceDetect=True, scale_factor=1.34)
            if not (face is None):
                face = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
                bb = dlib.rectangle(0, 0, 96, 96)
                face = prepData_Specific_pt2.align.align(96, face, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                face = np.around(face/255.0, decimals=12)
                add_imgs = add_imgs + addNoise(face, n)
                add_labels = add_labels + list(itertools.repeat(yLabel, n))
    add_imgs = np.array(add_imgs)
    add_labels = np.array(add_labels)
    print add_imgs.shape
    print add_labels.shape
    return np.append(X_data, add_imgs, axis=0), np.append(Y_data, add_labels, axis=0)






def process_image(img1):
    bb = dlib.rectangle(0, 0, 96, 96)
    img1 = img_to_encoding.align.align(96, img1, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    img = np.array([img])
    return img


def shuffle_in_unison(x, y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)


# load and shuffle data
def loadAndShuffle():
    X_data = np.load("xdata.npy")
    Y_data = np.load("ydata.npy")
    shuffle_in_unison(X_data, Y_data)
    return X_data, Y_data

# given input data, generate training and test sets (CV will be done in .fit)
def finishData(X_data, Y_data):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=.10)
    return X_train, X_test, Y_train, Y_test

# load data not in time series
def loadCNNData():
    X_data = np.load("xdata.npy")
    Y_data = np.load("ydata.npy")
    shuffle_in_unison(X_data, Y_data)
    X_data = X_data.reshape(-1, 96, 96, 3)
    Y_data = np.repeat(Y_data, 15)
    return finishData(X_data, Y_data)

def loadCNNData2():
    X_data, Y_data, _, _, _ = np.load("data.npy")
    shuffle_in_unison(X_data, Y_data)
    return finishData(X_data, Y_data)

# load data for time series
def loadTimeData():
    X_data, Y_data = loadAndShuffle()
    return train_test_split(X_data, Y_data, test_size=.10)
