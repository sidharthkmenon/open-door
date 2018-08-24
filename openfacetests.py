
# coding: utf-8

# In[90]:


import openface
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras import optimizers
import tensorflow as tf
import numpy as np
import cv2
import dlib
import time

# In[5]:



with CustomObjectScope({'tf': tf}):
    model = load_model('./nn4.small2.v1.h5')


# In[ ]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

# In[97]:


#from https://github.com/obieda01/Deep-Learning-Specialization-Coursera/blob/master/Course%204%20-%20Convolutional%20Neural%20Networks/Week%204/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20%20v1.ipynb
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.squared_difference(anchor, positive))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.squared_difference(anchor, negative))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    maxi = tf.maximum(basic_loss, 0.0)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss

# expands w&h by scale (from results in paper), and keeps same centroid
def fitToImg((x, y, w, h), (imgw, imgh, c), scale=1.0):
    nw, nh = map(lambda n: n*scale, (w,h))
    adjust = lambda nd, d: (nd/2) - (d/2)
    relu = lambda i: 0 if i < 0 else i
    nx, ny = relu(x - adjust(nw, w)), relu(y - adjust(nh, h))
    nw, nh = min((imgw-nx), nw), min((imgh-ny), nh)
    return nx, ny, nw, nh

# In[98]:


# rect = (400, 200, 100, 200)
# checkCentroid(rect, fitToImg(rect, (640, 480, 3), scale=1.8))

def checkCentroid(pt1, pt2):
    calc = lambda (g, h, j, k): (g + j/2, h + k/2)
    return (calc(pt1) == calc(pt2))

def toCVFormat2(face_list):
    newList = []
    for faces in face_list:
        x = faces.rect.left()
        y = faces.rect.top()
        newList.append((x, y, faces.rect.right() - x, faces.rect.bottom() - y))
    return newList

adam= optimizers.Adam()
model.compile(optimizer='adam', loss=triplet_loss, metrics = ['accuracy'])

# thank god for stackoverflow.
def histEqualize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def cnnDetect(img, faceDetect=True, equalize_hist=True, scale_factor=1.0):
    if equalize_hist:
        img = histEqualize(img)
    if faceDetect:
        face_list = cnn_face_detector(img, 1)
        face_boxes = toCVFormat2(face_list)
        face_areas = [w*h for x, y, w, h in face_boxes]
        if len(face_areas) != 0:
            x, y, w, h = fitToImg(face_boxes[face_areas.index(max(face_areas))],
                img.shape, scale=scale_factor)
            crop = img[int(y):int(y + h), int(x):int(x + w), :]
            return crop
        else:
            return None
    else:
        return img

def cropFace(img):
    img = histEqualize(img)
    face_list = cnn_face_detector(img, 1)
    face_boxes = toCVFormat2(face_list)
    face_areas = [w*h for x, y, w, h in face_boxes]
    if len(face_areas) != 0:
        x, y, w, h = fitToImg(face_boxes[face_areas.index(max(face_areas))],
            img.shape, scale=1.34)
        crop = img[int(y):int(y + h), int(x):int(x + w), :]
        return crop
    else:
        return None

def testHelp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = [w*h for x, y, w, h in face_boxes]
    if len(face_areas) != 0:
        x0, y0, w0, h0 = detect_largest_face(img)
        x, y, w, h = fitToImg(face_boxes[face_areas.index(max(face_areas))],
        img.shape, scale=1.34)
        return (int(x0), int(y0)), (int(x+w0), int(y+h0)), (int(x), int(y)), (int(x + w), int(y + h))
# In[118]:



def img_to_encoding(img_path, model):
    start = time.time()
    img1 = cv2.imread(img_path, 1)
    x, y, w, h = detect_largest_face(img1)
    print "Face detection took %s secs" % (time.time() - start)

    start = time.time()
    cv2.imshow('largest face', img1[y:y+h, x:x+w])
    cv2.waitKey()
    if img_to_encoding.align is None:
        facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
        img_to_encoding.align = openface.AlignDlib(facePredictor)
    print "Face alignment part 1 took %s secs" % (time.time() - start)
    s = time.time()
    bb = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    img1 = img_to_encoding.align.align(96, img1, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#     print "Face alignment part 2 took %s secs" % (time.time() - s)
    print "Face alignment took %s secs" % (time.time() - start)

    start = time.time()
    print "post encoding alignment is: {0}".format(img1.shape)
    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    print "Forward pass took %s secs" % (time.time() - start)
    return embedding

facePredictor = '/Users/sidharthmenon/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
img_to_encoding.align = openface.AlignDlib(facePredictor)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = [w*h for x, y, w, h in face_boxes]
    return face_boxes[face_areas.index(max(face_areas))]

# def get_cropped_face(img_path):
#     img = cv2.imread(img_path, 1)
#     #TODO: take out manual resizing completely
#     #
#     face_box = detect_largest_face(img)
#     img_cropped = img[y:y+h, x:x+w]
#     img_cropped = cv2.resize(img_cropped, (96, 96))
#     cv2.imshow('cropped', img_cropped)
#     cv2.waitKey()
#     return img_cropped, face_box


# In[119]:


database = {}
#TODO: uncomment
# database['ketan'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/ketan-1.jpg', model)
# database['sid'] = img_to_encoding('/Users/sidharthmenon/Desktop/Sid.jpeg', model)
# database['parker'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/parkerface.jpeg', model)
# database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)


# In[120]:


# print database['ketan']


# In[121]:



# GRADED FUNCTION: who_is_it# GRADED

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE ###

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    start = time.time()
    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding, ord=2)
        print "distance from photo to %s is %s" % (name, dist)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    print "Face identification took %s secs" % (time.time() - start)
    return min_dist, identity


# In[124]:


# start = time.time()
# who_is_it('/Users/sidharthmenon/Desktop/acha2.jpeg', database, model)
# # who_is_it('/Users/ketanagrawal/Desktop/image-test/ketan-5.jpg', database, model)
# end = time.time()
# print "Total time taken: %s" % (end - start)
