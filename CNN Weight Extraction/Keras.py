#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 28, 28

# number of channels
img_channels = 3

#%%
#  data

path1 = 'Warm'    #path of folder of images     
path2 = 'Cold'

listing1 = os.listdir(path1) 
listing2 = os.listdir(path2) 
num_samples1=size(listing1)
num_samples2=size(listing2)
print num_samples1
print num_samples2

imlist1 = os.listdir(path1)
imlist2 = os.listdir(path2)

im1 = array(Image.open('Cold' + '/'+ imlist2[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr1 = len(imlist1) # get the number of images
imnbr2 = len(imlist2)

# create matrix to store all flattened images
immatrix1 = array([array(Image.open('Warm'+ '/' + im3)).flatten()
              for im3 in imlist1],'f')
immatrix2 = array([array(Image.open('Cold'+ '/' + im4)).flatten()
              for im4 in imlist2],'f')

immatrix = np.concatenate((immatrix1,immatrix2))
                
label=np.ones((num_samples1+num_samples2,),dtype = int)
label[0:num_samples1]=0
label[num_samples1:num_samples2]=1

num_samples = num_samples1+num_samples2
print num_samples
print size(immatrix)
print size(label)

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]


#img=immatrix[167].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
#print (train_data[0].shape)
#print (train_data[1].shape)

#%%

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 1


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

#%%
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta' , metrics=["acc"])

#%%
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
            
            
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_split=0.2)


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])




#%%       

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])



# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)



# Loading weights

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)

