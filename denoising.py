import matplotlib.pyplot as plt
import numpy as np
import PIL
import keras.backend as K
from quaternion_layers import QConv2D as Convolution2D
from quaternion_layers import QDense as Dense
from keras.models import Sequential
from keras.models import Model
from keras.utils import multi_gpu_model
import random
#from keras.layers import Dense
#from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Add
from keras.optimizers import Adam
import os


encoding_size = 90
perturbation_max = 40
image_shape = (128,128,3)

preprocess = lambda x : x / 127 - 1
deprocess  = lambda x :((x + 1.) * 127.0).astype(np.uint8)

model_input = Input(shape = image_shape)
conv1       = Convolution2D(23, 3, padding = 'same', activation = 'relu')(model_input)
conv2       = Convolution2D(23, 3, padding = 'same', activation = 'relu')(conv1)
pool1       = AveragePooling2D()(conv2)
conv3       = Convolution2D(45, 3, padding = 'same', activation = 'relu')(pool1)
conv4       = Convolution2D(45, 3, padding = 'same', activation = 'relu')(conv3)
pool2       = AveragePooling2D()(conv4)
conv5       = Convolution2D(90, 3, padding = 'same', activation = 'relu')(pool2)
conv6       = Convolution2D(90, 3, padding = 'same', activation = 'relu')(conv5)
flatten     = Flatten()(conv6)
encoding    = Dense(encoding_size, activation = 'relu')(flatten)
dense2      = Dense(192, activation = 'relu')(encoding)
reshape     = Reshape((8, 8, 9))(dense2)
upsample2   = UpSampling2D(size = (4, 4))(reshape)
conv11      = Convolution2D(90, 3, padding = 'same', activation = 'relu')(upsample2)
conv12      = Convolution2D(90, 3, padding = 'same', activation = 'relu')(conv11)
add1        = Add()([conv12, conv6])
upsample3   = UpSampling2D()(add1)
conv13      = Convolution2D(45, 3, padding = 'same', activation = 'relu')(upsample3)
conv14      = Convolution2D(45, 3, padding = 'same', activation = 'relu')(conv13)
add2        = Add()([conv14, conv4])
upsample3   = UpSampling2D()(add2)
conv15      = Convolution2D(8, 3, padding = 'same', activation = 'relu')(upsample3)
conv16      = Convolution2D(1, 3, padding = 'same', activation = 'tanh')(conv15)

autoencoder = Model(model_input, conv16)
#autoencoder = multi_gpu_model(autoencoder1, gpus=3)
#autoencoder.load_weights('checkpointq/weight2999.h5')
Img = sorted(os.listdir('dataset'))
Val = sorted(os.listdir('validation_split'))
import scipy
from PIL import Image
import math
import skimage
def loadImage(batch_path):
    true = []
    noise = []
    for f in batch_path:
        img = Image.open(f)
        width = img.size[0]
        height = img.size[1]
        w = random.randint(0,width-129)
        h = random.randint(0,height-129)
        img = img.crop((w,h,(w+128),(h+128)))
        tmp = preprocess(np.array(img,dtype=np.float32))
        tmp[tmp>1.] = 1.0
        ntp = skimage.util.random_noise(tmp,mode='s&p',amount=0.3)
        ntp = skimage.util.random_noise(ntp,mode='gaussian')
        true.append(tmp)
        noise.append(ntp)
    return np.array(true),np.array(noise)

def loadValImage(batch_path):
    true = []
    noise = []
    for f in batch_path:
        img = Image.open(f)
        tmp = preprocess(np.array(img,dtype=np.float32))
        tmp[tmp>1.] = 1.0
        ntp = skimage.util.random_noise(tmp,mode='s&p',amount=0.3)
        ntp = skimage.util.random_noise(ntp,mode='gaussian')
        true.append(tmp)
        noise.append(ntp)
    return np.array(true),np.array(noise)

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = \
            img[:, :, :]
    return image

def PSNR(im1s,im2s):
    n = len(im1s)
    psnr = 0
    for i in range(n):
        im1si = im1s[i]*127.5 +127.5
        im2si = im2s[i]*127.5 +127.5
        diff = (im1si - im2si)**2
        mse = np.mean(diff)
        psnr += 10*np.log10(255*255/mse)
    return psnr/n

from skimage.viewer import ImageViewer
epochs = 3000
batch_size=32
test_size=64
opt = Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-8)
#opt = SGD(lr = 0.001, momentum=0.9, decay=4e-5)
autoencoder.compile(loss='mse',optimizer=opt,metrics=['accuracy'])
maxpsnr = 30.0
#learning_rate = 0.001
for e in range(epochs):
    batch_counter = 0
    test_counter = 0
    psnr_test = 0
    psnr_mtr = 0
    while batch_counter * batch_size < 3500:
        path_true = Img[batch_counter*batch_size:min((batch_counter+1)*batch_size,3500)]
        path_true = ['dataset/'+i for i in path_true]
        true, noise = loadImage(path_true)
        loss = autoencoder.train_on_batch(noise, true)
        samples = autoencoder.predict(noise)
        psnr_mtr += PSNR(true, samples)
        batch_counter += 1
        '''
        if not (batch_counter % 50) and batch_counter:
            image = combine_images(samples)
            image = image*127.5 +127.5
            Image.fromarray(image.astype(np.uint8)).save('outputq1/'+str(e)+'_'+str(batch_counter)+'.jpg')
            '''
    psnr_mtr = psnr_mtr/batch_counter
    print '[epoch%s] train loss:%s psnr:%s'%((e), loss, psnr_mtr)
    while test_counter * test_size+3500 < len(Img):
        path_true = Val[test_counter*test_size:min((test_counter+1)*test_size,1000)]
        path_true = ['validation_split/'+i for i in path_true]
        true, noise = loadValImage(path_true)
        #loss = autoencoder.train_on_batch(noise, true)
        samples = autoencoder.predict(noise)
        psnr_test += (PSNR(true, samples)*len(path_true))
        #print '[epoch%s|] test1 psnr:%s'%(e, psnr_mtr)
        test_counter += 1
        '''
        if not (test_counter % 10) and test_counter:
            image = combine_images(samples)
            image = image*127.5 +127.5
            Image.fromarray(image.astype(np.uint8)).save('output_testq1/'+str(e)+'_'+str(test_counter)+'.jpg')
            '''
    psnr_test = psnr_test/1000
    print '[epoch%s|] test psnr:%s'%((e), psnr_test)
    if psnr_test>maxpsnr:
        maxpsnr = psnr_test
        autoencoder.save_weights('checkpointq/best.h5')
    if not ((e+1)%100):
        autoencoder.save_weights('checkpointq/weight%s.h5'%(e))
    #if K.get_value(opt.lr)>0.5:
        #K.set_value(opt.lr, 0.998 * K.get_value(opt.lr))


