
# coding: utf-8

# In[1]:

import io
import openface
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras import optimizers
import tensorflow as tf
import numpy as np
import cv2
import dlib
import time
import picamera

# In[5]:


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


# In[98]:


# In[118]:


def img_to_encoding(img_path, model):
    start = time.time()
    img1 = cv2.imread(img_path, 1)
    x, y, w, h = detect_largest_face(img1)
    if (x, y, w, h) == (0, 0, 0, 0):
        return None
    print "Face detection took %s secs" % (time.time() - start)
    
    start = time.time()
#     cv2.imshow('largest face', img1[y:y+h, x:x+w])
#     cv2.waitKey()
#     if img_to_encoding.align is None:
#         facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
#         img_to_encoding.align = openface.AlignDlib(facePredictor)
#     print "Face alignment part 1 took %s secs" % (time.time() - start)
#     s = time.time()
    bb = dlib.rectangle(x, y, x + w, y + h)
    img1 = img_to_encoding.align.align(96, img1, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
#     print "Face alignment part 2 took %s secs" % (time.time() - s)
    print "Face alignment took %s secs" % (time.time() - start)
    
    start = time.time()
    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    print "Forward pass took %s secs" % (time.time() - start)
    return embedding

facePredictor = '/home/pi/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
img_to_encoding.align = openface.AlignDlib(facePredictor)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = detect_largest_face.face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = [w*h for x, y, w, h in face_boxes]
    if len(face_areas) == 0:
        return 0, 0, 0, 0
    return face_boxes[face_areas.index(max(face_areas))]

detect_largest_face.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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


# In[120]:

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
    if encoding is None:
        print "Face not found. Taking another picture"
        return False
#    start = time.time()
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
        return False
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        return True
#    print "Face identification took %s secs" % (time.time() - start)    
    #return min_dist, identity


# In[124]:

if __name__ == "__main__":
    adam= optimizers.Adam()
    with CustomObjectScope({'tf': tf}):
        model = load_model('./nn4.small2.v1.h5')
    model.compile(optimizer='adam', loss=triplet_loss, metrics = ['accuracy'])
    database = {}
    database['ketan'] = img_to_encoding('/home/pi/image-test/ketan-1.jpg', model)
    database['sid'] = img_to_encoding('/home/pi/image-test/sid-1.jpeg', model)
    database['parker'] = img_to_encoding('/home/pi/image-test/parkerface.jpeg', model)
    database['aditya'] = img_to_encoding('/home/pi/image-test/aditya-1.jpg', model)
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    time.sleep(2)
    face_found = False
    while not face_found:
        stream = io.BytesIO()
        camera.capture('foo.jpg')
        face_found = who_is_it('foo.jpg', database, model)
    #start = time.time()
    #who_is_it('/home/pi/ketan-picam.jpg', database, model)
    # who_is_it('/Users/ketanagrawal/Desktop/image-test/ketan-5.jpg', database, model)
    #end = time.time()
    #print "Total time taken: %s" % (end - start)

