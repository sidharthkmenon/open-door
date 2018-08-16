import dlib
import cv2
import numpy as np
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='/Users/sidharthmenon/Desktop/sid-2.jpeg')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='/Users/sidharthmenon/Desktop/Summer\ 2018/open-door')
args = ap.parse_args()

hog_detect = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

def toCVFormat(face_list):
    newList = []
    for faces in face_list:
        x = faces.left()
        y = faces.top()
        newList.append((x, y, faces.right() - x, faces.bottom() - y))
    return newList

def toCVFormat2(face_list):
    newList = []
    for faces in face_list:
        x = faces.rect.left()
        y = faces.rect.top()
        newList.append((x, y, faces.rect.right() - x, faces.rect.bottom() - y))
    return newList

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = [w*h for x, y, w, h in face_boxes]
    return face_boxes[face_areas.index(max(face_areas))]


# expands w&h by scale (from results in paper), and keeps same centroid
def fitToImg((x, y, w, h), (imgw, imgh, c), scale=1.0):
    nw, nh = map(lambda n: n*scale, (w,h))
    adjust = lambda nd, d: (nd/2) - (d/2)
    relu = lambda i: 0 if i < 0 else i
    nx, ny = relu(x - adjust(nw, w)), relu(y - adjust(nh, h))
    nw, nh = min((imgw-nx), nw), min((imgh-ny), nh)
    return nx, ny, nw, nh

# thank god for stackoverflow.
def histEqualize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# init images
X_data, Y_data, NoFaceData, NoFaceData_Labels = np.load("data.npy")
imgs = random.sample(X_data, 20)
nfImgs = random.sample(NoFaceData, 20)
testarray = []
for i in range(20):
    # img = nfImgs[i]
    # norm = histEqualize(img)
    # testarray.append(img)
    # testarray.append(norm)
    img = imgs[i]
    norm = histEqualize(img)
    testarray.append(img)
    testarray.append(norm)

print "testarray prepped"

# convert input array to output format for analysis
def toFormat(x):
    total = float(sum(x))
    nameField = lambda i: "CNN" if i == 0 else ("HOG" if i == 1 else ("Haar" if i == 2 else "Other"))
    for i in range(len(x)):
        num = x[i]
        x[i] = "{0}: {1} ({2}%)".format(nameField(i), num, 100*round(num/total, 2))
    return x


# return CNN/HOG/Haar/Other split for normalized and not normalized images
def splitImgResults(inputImgs):
    isNorm = False
    out = [0, 0, 0, 0]
    normOut = [0, 0, 0, 0]
    imType = lambda b: "normalized img" if b else "img"
    for img in inputImgs:
        cv2.imshow(imType(isNorm), img)
        key = cv2.waitKey(0)
        if not(isNorm):
            if key == 114: # red
                out[0] += 1
            elif key == 103: # green
                out[1] += 1
            elif key == 98: # blue
                out[2] += 1
            else:
                out[3] += 1
        else:
            if key == 114: # red
                normOut[0] += 1
            elif key == 103: # green
                normOut[1] += 1
            elif key == 98: # blue
                normOut[2] += 1
            else:
                normOut[3] += 1
        cv2.destroyAllWindows()
        isNorm = not(isNorm)
    return toFormat(out), toFormat(normOut)



# draw rectangles, if ID'd. green: HOG, red: CNN, blue: Haar
def test():
    # path = '/Users/sidharthmenon/Desktop/sid-fake.jpeg'
    # img = cv2.imread(path, 1)
    totalImgLength = len(testarray)
    currentImgNum = 1
    for img in testarray:
        face_list = hog_detect(img, 1)
        face_boxes = toCVFormat(face_list)
        face_areas = [w*h for x, y, w, h in face_boxes]
        if not (len(face_areas) == 0):
            x, y, w, h = face_boxes[face_areas.index(max(face_areas))]
            x, y, w, h = fitToImg((x, y, w, h), img.shape, scale=1.34)
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_list = cnn_face_detector(img, 1)
        face_boxes = toCVFormat2(face_list)
        face_areas = [w*h for x, y, w, h in face_boxes]
        if not (len(face_areas) == 0):
            x, y, w, h = face_boxes[face_areas.index(max(face_areas))]
            x, y, w, h = fitToImg((x, y, w, h), img.shape, scale=1.34)
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_areas = [w*h for x, y, w, h in face_boxes]
        if not (len(face_areas) == 0):
            x0, y0, w0, h0 = face_boxes[face_areas.index(max(face_areas))]
            x0, y0, w0, h0 = fitToImg((x0, y0, w0, h0), img.shape, scale=1.34)
            x0, y0, w0, h0 = int(x0), int(y0), int(w0), int(h0)
            cv2.rectangle(img, (x0, y0), (x0+h0, y0+w0), (255, 0, 0), 3)
        print "image {0} of {1} processed".format(currentImgNum, totalImgLength)
        currentImgNum += 1
    out, normOut = splitImgResults(testarray)
    print "best face detector test results below:"
    print "image results: {0}".format(out)
    print "normalized image results: {0}".format(normOut)


test()
