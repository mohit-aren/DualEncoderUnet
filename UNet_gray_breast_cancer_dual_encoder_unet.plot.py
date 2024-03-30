import numpy as np
import pandas as pd

import json
import sys
from PIL import Image, ImageOps

#from skimage.io import imread
#from matplotlib import pyplot as plt
import random

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] ='mode=FAST_RUN,device=cpu'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, ZeroPadding2D
from tensorflow.keras.layers import Activation, Flatten, Reshape
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from tensorflow.keras.utils import np_utils
#from tensorflow.keras.applications import imagenet_utils

import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout

import numpy as np
np.random.seed(1647)
#import sys
#sys.path.append('C:/Program Files (x86)/Graphviz2.38/bin/')
#from tensorflow.keras.utils import plot_model

smooth = 1.
dropout_rate = 0.5

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


########################################
# 2D Standard
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'elu'

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

########################################
path = 'results/'
img_w = 320
img_h = 320
n_labels = 2

Lung = [255,255,255]

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]


n_train = 51
n_test = 45
n_val = 10

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def label_map1(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            #print(labels[r][c])
            if(labels[r][c][0] >180 and labels[r][c][1] > 180 and labels[r][c][2] > 180):
                label_map[r, c, 1] = 1
            elif(labels[r][c][0] < 50 and labels[r][c][1] < 50 and labels[r][c][2] < 50):
                label_map[r, c, 0] = 1
    return label_map


import os
import imageio
def prep_data1(mode, autoencoder):
    data = []
    data1 = []
    label = []
    
    folder_path = 'Cropped10' # path + mode
    folder_path1 = 'XGray' # path + mode

    images_path = os.listdir(folder_path)

    if(mode == 'train'):
        n = 51
    elif(mode == 'val'):
        n = 45
    else:
        n = 10
    for index, image in enumerate(images_path):

        filename = os.path.join(folder_path, image)
    
        print(index, filename)
        if(index > 50 and mode == 'train'):
            continue
        elif((index < 51 or index > 95) and mode =='val'):
            continue
        elif((index < 96 or index > 105) and mode == 'test'):
            continue
        
        #truth_file = filename.split('.png')
        imgs = image.split('.')
        tfile = os.path.join(folder_path1, imgs[0] + '.jpg')
    
        
        print(tfile)
        if(filename == ""):
            break
        
        #img1 = Image.open(filename)
        img1 = Image.open(filename)
        w, h = img1.size
        start = (w-320)//2
        s_h = (h-320)//2
        
        new_im = img1.crop((start, s_h, start+320, s_h+320))
        #new_im.save('original_'+ imgs[0] + '.jpg')
        
        temp = []
        
        temp.append(np.reshape(new_im,(320, 320,3)))
        output = autoencoder.predict(np.array(temp), verbose=1)
        #stop = time.time()
        
    

        #print(stop-start)
        
        #print(output)
        output = output.reshape((output.shape[0], img_w, img_h, 2))
        #print(output[0])
        #print(test_label.shape())
        Lung = [255,255,255]
        Unlabelled = [0,0,0]
        
        labeled = np.argmax(output[0], axis=-1)
        #print(labeled)
        labeled1 = np.zeros([img_w, img_h, 3]) 
        for i in range(0,img_w):
            for j in range(0, img_h):
                if(labeled[i,j] == 1):
                    labeled1[i,j] = Lung
                else:
                    labeled1[i,j] = Unlabelled

        #imageio.imwrite('predict_unet_b_'+ imgs[0] + '.jpg', labeled1.astype('uint8'))
        #new_im = Image.new("RGB", (544, 512))
        #new_im.paste(img1, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))



        img2 = Image.open(tfile)
        #new_im1 = img2.resize((320, 320))
        new_im1 = img2.convert('RGB')
        new_im1 = new_im1.crop((start, s_h, start+320, s_h+320))
        #new_im1.save('GT_'+ imgs[0] + '.jpg')
        
        # create a new image and paste the resized on it
        
        #new_im1 = Image.new("RGB", (544, 512))
        #new_im1.paste(img2, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))


        #index += 1
        # create a new image and paste the resized on it
        

        #img, gt = [imread(path + mode + '/' + filename + '.png')], imread(path + mode + '-colormap/' + filename + '.png')
        
        img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        data.append(np.reshape(img,(320, 320,3)))
        data1.append(labeled1)
        label.append(label_map1(gt))
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, data1, label = np.array(data), np.array(data1), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    #print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048544, label.nbytes / 1048544))

    return data, data1, label




def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(np.reshape(img,(256,256,1)))
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048544, label.nbytes / 1048544))

    return data, label
"""
def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.JPG')
    plt.show()
"""

#########################################################################################################

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


#import skimage.io as io
#import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def get_unet():
        inputs = Input((320, 320, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(2, 1, activation='softmax')(conv9)
        conv10 = Reshape((320*320 , 2))(conv10)

        model = Model(inputs, conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        return model


def get_unet_multi():
        inputs = Input((320, 320, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)



        y = Model(inputs, drop5)
        
        inputs_1 = Input((320, 320, 3))

        conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs_1)
        # print(conv1)
        conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
        pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

        conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_1)
        conv2_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
        pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

        conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_1)
        conv3_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)
        pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

        conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_1)
        conv4_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_1)
        drop4_1 = Dropout(0.5)(conv4_1)
        pool4_1 = MaxPooling2D(pool_size=(2, 2))(drop4_1)

        conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_1)
        conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_1)
        drop5_1 = Dropout(0.5)(conv5_1)

        x = Model(inputs_1, drop5_1)

        combined = Lambda(lambda a: a[0] + a[1])([x.output, y.output])
        #combined = tf.math.sum([x.output, y.output])
        
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(combined))
        merge6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        
        conv10 = Conv2D(2, 1, activation='softmax')(conv9)
        conv10 = Reshape((320*320 , 2))(conv10)
        
        model = Model([inputs, inputs_1], conv10)
        
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        # combine the output of the two branches
        return model


"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder1 = get_unet()
autoencoder = get_unet_multi()
#plot_model(autoencoder, to_file='model.png', show_shapes=True)

print('Start')
#optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
#autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights
autoencoder1.load_weights('model_5l_weight_lung_unet.1.hdf5')

#train_data, train_data1, train_label = prep_data1('train',autoencoder1)
#val_data, val_data1, val_label = prep_data1('val',autoencoder1)
nb_epoch = 10
batch_size = 2
#history = autoencoder.fit([train_data, train_data1], train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=([val_data, val_data1], val_label))

autoencoder.load_weights('model_5l_weight_lung_unet_multi.hdf5')

#3Lt1.Crop10
images = ['27Rt1.Crop10.jpg','31Rt1.Crop10.jpg', '48Lt1.Crop10.jpg', '55Lt1.Crop10.jpg']
folder_path = 'Cropped10' # path + mode
folder_path1 = 'XGray' # path + mode

for image in images:

        filename = os.path.join(folder_path, image)
    
        imgs = image.split('.')
        #img1 = Image.open(filename)
        img1 = Image.open(filename)
        w, h = img1.size
        start = (w-320)//2
        s_h = (h-320)//2
        
        new_im = img1.crop((start, s_h, start+320, s_h+320))
        #new_im.save('original_'+ imgs[0] + '.jpg')
        
        temp = []
        
        temp.append(np.reshape(new_im,(320, 320,3)))
        output = autoencoder1.predict(np.array(temp), verbose=1)
        #stop = time.time()
        
    

        #print(stop-start)
        
        #print(output)
        output = output.reshape((output.shape[0], img_w, img_h, 2))
        #print(output[0])
        #print(test_label.shape())
        Lung = [255,255,255]
        Unlabelled = [0,0,0]
        data1 = []
        labeled = np.argmax(output[0], axis=-1)
        #print(labeled)
        labeled1 = np.zeros([img_w, img_h, 3]) 
        for i in range(0,img_w):
            for j in range(0, img_h):
                if(labeled[i,j] == 1):
                    labeled1[i,j] = Lung
                else:
                    labeled1[i,j] = Unlabelled
                    
        data1.append(labeled1)
        tfile = os.path.join(folder_path1, imgs[0] + '.jpg')
    
        
        print(tfile)
        if(filename == ""):
            break
        
        #img1 = Image.open(filename)
        img1 = Image.open(filename)
        w, h = img1.size
        start = (w-320)//2
        s_h = (h-320)//2
        
        new_im = img1.crop((start, s_h, start+320, s_h+320))
        #new_im.save('original_'+ imgs[0] + '.jpg')
        
        temp = []
        
        temp.append(np.reshape(new_im,(320, 320,3)))
        output = autoencoder.predict([np.array(temp), np.array(data1)], verbose=1)
        output = output.reshape((output.shape[0], img_w, img_h, 2))
        #stop = time.time()
        
    

        #print(stop-start)
        
        

        #imageio.imwrite('predict_unet_b_'+ imgs[0] + '.jpg', labeled1.astype('uint8'))
        #new_im = Image.new("RGB", (544, 512))
        #new_im.paste(img1, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))


        print(output)
        labeled = np.argmax(output[0], axis=-1)
        print(labeled)
        #print(labeled)
        labeled1 = np.zeros([img_w, img_h, 3]) 
        for i in range(0,img_w):
            for j in range(0, img_h):
                if(labeled[i,j] == 1):
                    labeled1[i,j] = Lung
                else:
                    labeled1[i,j] = Unlabelled
        imageio.imwrite('predict_unet_complex_'+ imgs[0] + '.jpg', labeled1.astype('uint8'))
 