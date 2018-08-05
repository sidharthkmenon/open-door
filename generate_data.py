import numpy as np
import cv2
import os
from os import listdir

projectPath = "/Users/sidharthmenon/Desktop/Summer 2018/open-door/liveness-dataset/"


target = projectPath + "7/Down/Vl_NT_HS_wg_E_7_120.mp4"
target2 = projectPath + "7/Left/G_NT_HS_wg_E_7_6.mp4"
target3 = projectPath + "7/Right/G_NT_5s_wg_E_7_1.mp4"



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

def prepData():
    X_data = []
    Y_data = []
    yEval = lambda s: 1 if (s[0] == "G") else 0
    for index in map(str, [7, 9, 10, 14, 15, 18, 20, 23]):
        for dir in ["Up", "Down", "Left", "Right"]:
            folderPath = projectPath + index + "/" + dir
            for fileName in os.listdir(folderPath):
                filePath = folderPath + "/" + fileName
                vid = cv2.VideoCapture(filePath)
                res = []
                frameNum = 0
                while(frameNum < 30):
                    frameNum = frameNum + 1
                    _, frame = vid.read()
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

X_data, Y_data = prepData()
np.save("xdata.npy", X_data)
np.save("ydata.npy", Y_data)
print "saved"
