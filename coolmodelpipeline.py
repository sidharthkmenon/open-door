
# coding: utf-8

# In[78]:


import openface
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras import optimizers
import keras
from keras.layers import Input, Activation, Dropout, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import cv2
import dlib
import time
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from generate_data import prepData_Specific, prepData_Specific_pt1, prepData_Specific_pt2, augmentData, augmentData2
import resource
from subprocess import call
# In[55]:






with CustomObjectScope({'tf': tf}):
    model = load_model('./nn4.small2.v1.h5')


# In[ ]:


print model.summary()


# In[39]:


type(model.layers[163].output)


# In[48]:


get_flatten_layer_output = K.function([model.layers[0].input], [model.layers[163].output])
# start = time.time()
img1 = cv2.imread('/Users/ketanagrawal/Desktop/image-test/parkerface.jpeg', 1)
bb = detect_largest_face2(img1)
#     x, y, w, h = detect_largest_face(img1)
# print "Face detection took %s secs" % (time.time() - start)

# start = time.time()
#     cv2.imshow('largest face', img1[y:y+h, x:x+w])
#     cv2.waitKey()
#     if img_to_encoding.align is None:


# In[68]:


def process_image(img1):
    bb = dlib.rectangle(0, 0, 96, 96)
    img1 = img_to_encoding.align.align(96, img1, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    img = np.array([img])
    return img


# In[74]:


_, hdata_lim = resource.getrlimit(resource.RLIMIT_DATA)
resource.setrlimit(resource.RLIMIT_DATA, (hdata_lim, hdata_lim))
_, hfile_lim = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (hfile_lim, hfile_lim))
def do_it():
    print "loading data"
    X_data, Y_data, _, _ = prepData_Specific('PigLatin2_v2', first_resize=(150,150), histEqualize=True, faceDetect=True, scale=1.34, align=True)
    print "getting intermediate layer"
    intermediate_output = np.asarray([get_flatten_layer_output([x]) for x in X_data])
    return intermediate_output, Y_data


# In[75]:


def coolModel(input_shape):
    X_input = Input(input_shape)
    model = Dense(256)(X_input)
    model = Activation('relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(16, activation='relu')(model)
    model = Dropout(0.125)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=X_input, outputs=model, name='cool model')
    return model

def coolModel_v2(input_shape):
    X_input = Input(input_shape)
    model = Dense(256)(X_input)
    model = Activation('relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=X_input, outputs=model, name='cool model')
    return model

def coolModel_v2_5(input_shape):
    X_input = Input(input_shape)
    model = Dense(128, activation='relu')(X_input)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=X_input, outputs=model, name='cool model')
    return model

def coolModel_v2_6(input_shape):
    X_input = Input(input_shape)
    model = Dropout(0.5)(X_input)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=X_input, outputs=model, name='cool model')
    return model

def coolModel_v3(input_shape):
    X_input = Input(input_shape)
    model = Dense(256)(X_input)
    model = Activation('relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=X_input, outputs=model, name='cool model')
    return model

# In[76]:


def shuffle(x, y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

def finishData(X_data, Y_data):
    print "shuffling"
    shuffle(X_data, Y_data)
    print "splitting"
    return train_test_split(X_data, Y_data, test_size=.10)
    print "done"


# In[106]:


# X_train = np.array(X_train)
# print X_train.shape


# In[107]:


# Y_train = np.array(Y_train)
# print Y_train.shape


# In[110]:

#
# X_test = np.array([x[0][0] for x in X_test])
# Y_test = np.array([x[0] for x in Y_test])
# print X_test.shape
# print Y_test.shape

# TODO: Delete This
# prepData_Specific_pt1(nickName, first_resize=None):
# prepData_Specific_pt2(fileName=None, rx=None, ry=None, histEqualize=True, faceDetect=True, scale=1.0, align=False):

# In[112]:



NoAlign = coolModel_v2((736,))
NoAlign.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print "loading data pt1"
rawData, rawLabels = np.load('./150_Data/150_Imgs.npy'), np.load('./150_Data/150_Labels.npy')
print rawData.shape, rawLabels.shape
nickName = '150-v4'
call(['mkdir', '{0}'.format(nickName)])
print 'prepping Data'
X_data, Y_data, NoFaceData, NoFaceData_Labels = prepData_Specific_pt2(rx=rawData, ry=rawLabels, histEqualize=True, faceDetect=True, scale=1.34, align=True)
X_data, Y_data, NoFaceData, NoFaceData_Labels = map(np.array, (X_data, Y_data, NoFaceData, NoFaceData_Labels))
print X_data.shape, Y_data.shape
print NoFaceData.shape, NoFaceData_Labels.shape
X_data, Y_data = augmentData(X_data, Y_data, 50, nickName)
add_imgs, add_labels = augmentData2(50, .15, nickName)
X_data, Y_data = np.append(X_data, add_imgs, axis=0), np.append(Y_data, add_labels, axis=0)
np.save('{0}_Imgs.npy'.format(nickName), X_data)
np.save('{0}_Labels.npy'.format(nickName), Y_data)
np.save('{0}_NoFaceImgs.npy'.format(nickName), NoFaceData)
np.save('{0}_NoFaceLabels.npy'.format(nickName), NoFaceData_Labels)
call(['mv', './{0}_Imgs.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_Labels.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_NoFaceImgs.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_NoFaceLabels.npy'.format(nickName), './{0}'.format(nickName)])
X_data = np.load('./{0}/{0}_Imgs.npy'.format(nickName))
Y_data = np.load('./{0}/{0}_Labels.npy'.format(nickName))
print "getting intermediate layer"
intermediate_output = np.array([np.reshape(x, (-1, 96, 96, 3)) for x in X_data])
intermediate_output = np.array([get_flatten_layer_output([x]) for x in intermediate_output])
intermediate_output.shape
intermediate_output = np.reshape(intermediate_output, (intermediate_output.shape[0], 736))
Y_data.shape
intermediate_output.shape
X_train, X_test, Y_train, Y_test = finishData(intermediate_output, Y_data)
print "finished data"
cb = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2)]
simple_history = NoAlign.fit(X_train, Y_train, batch_size=32, epochs=30, validation_split=.15, callbacks=cb, verbose=1)
print(simple_history.history.keys())
plt.plot(simple_history.history['acc'])
plt.plot(simple_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(simple_history.history['loss'])
plt.plot(simple_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_performance = NoAlign.evaluate(X_test, Y_test)
print test_performance
NoAlign.save('{0}_Model.h5'.format(nickName))
call(['mv', './{0}_model.h5'.format(nickName), './{0}'.format(nickName)])


NoAlign = coolModel_v2((736,))
NoAlign.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nickName = '150-v4-random'
call(['mkdir', '{0}'.format(nickName)])
rawData, rawLabels = np.load('./150_Data/150_Imgs.npy'), np.load('./150_Data/150_Labels.npy')
print rawData.shape, rawLabels.shape
print 'prepping Data'
X_data, Y_data, NoFaceData, NoFaceData_Labels = prepData_Specific_pt2(rx=rawData, ry=rawLabels, histEqualize=True, faceDetect=True, scale=1.34, align=True, randomize=True)
X_data, Y_data, NoFaceData, NoFaceData_Labels = map(np.array, (X_data, Y_data, NoFaceData, NoFaceData_Labels))
print X_data.shape, Y_data.shape
print NoFaceData.shape, NoFaceData_Labels.shape
X_data, Y_data = augmentData(X_data, Y_data, 50, nickName)
add_imgs, add_labels = augmentData2(50, .15, nickName)
X_data, Y_data = np.append(X_data, add_imgs, axis=0), np.append(Y_data, add_labels, axis=0)
np.save('{0}_Imgs.npy'.format(nickName), X_data)
np.save('{0}_Labels.npy'.format(nickName), Y_data)
np.save('{0}_NoFaceImgs.npy'.format(nickName), NoFaceData)
np.save('{0}_NoFaceLabels.npy'.format(nickName), NoFaceData_Labels)
call(['mv', './{0}_Imgs.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_Labels.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_NoFaceImgs.npy'.format(nickName), './{0}'.format(nickName)])
call(['mv', './{0}_NoFaceLabels.npy'.format(nickName), './{0}'.format(nickName)])
X_data = np.load('./{0}/{0}_Imgs.npy'.format(nickName))
Y_data = np.load('./{0}/{0}_Labels.npy'.format(nickName))
print "getting intermediate layer"
intermediate_output = np.array([np.reshape(x, (-1, 96, 96, 3)) for x in X_data])
intermediate_output = np.array([get_flatten_layer_output([x]) for x in intermediate_output])
intermediate_output.shape
intermediate_output = np.reshape(intermediate_output, (intermediate_output.shape[0], 736))
Y_data.shape
intermediate_output.shape
X_train, X_test, Y_train, Y_test = finishData(intermediate_output, Y_data)
print "finished data"
cb = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2)]
simple_history = NoAlign.fit(X_train, Y_train, batch_size=32, epochs=30, validation_split=.15, callbacks=cb, verbose=1)
print(simple_history.history.keys())
plt.plot(simple_history.history['acc'])
plt.plot(simple_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(simple_history.history['loss'])
plt.plot(simple_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_performance = NoAlign.evaluate(X_test, Y_test)
print test_performance
NoAlign.save('{0}_Model.h5'.format(nickName))
call(['mv', './{0}_model.h5'.format(nickName), './{0}'.format(nickName)])

# In[113]:


# In[25]:


# Train PigLatin (best model so far) on 20-30 more epochs

nickName = 'PigLatin'
print 'loading data...'
X_data = np.load('./{0}/{0}_Imgs.npy'.format(nickName))
Y_data = np.load('./{0}/{0}_Labels.npy'.format(nickName))
X_data, Y_data = generate_data.augmentData(X_data, Y_data, 85)
print X_data.shape
print Y_data.shape
intermediate_output = np.array([np.reshape(x, (-1, 96, 96, 3)) for x in X_data])
print 'converting data to encoding...'
intermediate_output = np.array([get_flatten_layer_output([x]) for x in intermediate_output])
intermediate_output = np.reshape(intermediate_output, (intermediate_output.shape[0], 736))
print intermediate_output.shape, Y_data.shape
X_train, X_test, Y_train, Y_test = finishData(intermediate_output, Y_data)
np.save('intermediate_encoding2.npy', intermediate_output)
np.save('intermediate_labels2.npy', Y_data)
coolmodel = coolModel_v2((736,))
coolmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cb = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2)]
simple_history = coolmodel.fit(X_train, Y_train, batch_size=32, epochs=30, validation_split=.15, callbacks=cb, verbose=1)
print(simple_history.history.keys())
plt.plot(simple_history.history['acc'])
plt.plot(simple_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(simple_history.history['loss'])
plt.plot(simple_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_performance = coolmodel.evaluate(X_test, Y_test)
print test_performance
print 'saving...'
coolmodel.save('coolmodelv8_model.h5')
call(['mv', './coolmodelv8_model.h5', './coolmodelv3'])

#


with CustomObjectScope({'tf': tf}):
    cModel = load_model('./coolmodelv3/coolmodelv8_model.h5')

X_test = np.append(np.load('./coolmodelv3/add_imgs.npy'), np.load('more_imgs.npy'), axis=0)
Y_test = np.append(np.load('./coolmodelv3/add_labels.npy'), np.load('more_labels.npy'), axis=0)
print X_test.shape, Y_test.shape
intermediate_encoding = np.array([np.reshape(x, (-1, 96, 96, 3)) for x in X_test])
intermediate_encoding = np.array([get_flatten_layer_output([x]) for x in intermediate_encoding])
print intermediate_encoding.shape
intermediate_encoding = np.reshape(intermediate_encoding, (intermediate_encoding.shape[0], 736))

cModel.evaluate(x=intermediate_encoding, y=Y_test)


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


# In[10]:


adam= keras.optimizers.Adam()
model.compile(optimizer='adam', loss=triplet_loss, metrics = ['accuracy'])


# In[24]:


def img_to_encoding(img_path, model):
    start = time.time()
    img1 = cv2.imread(img_path, 1)
    bb = detect_largest_face2(img1)
#     x, y, w, h = detect_largest_face(img1)
    print "Face detection took %s secs" % (time.time() - start)

    start = time.time()
#     cv2.imshow('largest face', img1[y:y+h, x:x+w])
#     cv2.waitKey()
#     if img_to_encoding.align is None:
#         facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
#         img_to_encoding.align = openface.AlignDlib(facePredictor)
#     print "Face alignment part 1 took %s secs" % (time.time() - start)
#     s = time.time()
#     bb = dlib.rectangle(x, y, x + w, y + h)
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

facePredictor = '/Users/ketanagrawal/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
img_to_encoding.align = openface.AlignDlib(facePredictor)

def detect_largest_face2(img):
    return img_to_encoding.align.getLargestFaceBoundingBox(img)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = detect_largest_face.face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = [w*h for x, y, w, h in face_boxes]
    x, y, w, h = face_boxes[face_areas.index(max(face_areas))]
    return dlib.rectangle(x, y, x + w, y + h)

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


# In[34]:


database = {}
database['ketan'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/ketan-1.jpg', model)
database['sid'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/sid-1.jpeg', model)
database['parker'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/parkerface.jpeg', model)
database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)
database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)
database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)
database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)
database['aditya'] = img_to_encoding('/Users/ketanagrawal/Desktop/image-test/aditya-1.jpg', model)


# In[26]:


print database['ketan']


# In[27]:



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


# In[28]:


start = time.time()
who_is_it('/Users/ketanagrawal/Desktop/image-test/ketan-2.jpg', database, model)
# who_is_it('/Users/ketanagrawal/Desktop/image-test/ketan-5.jpg', database, model)
end = time.time()
print "Total time taken: %s" % (end - start)
