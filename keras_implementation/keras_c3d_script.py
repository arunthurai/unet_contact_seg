#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!cp /home/athurai3/scratch/test_preproc_contact_loc/train_patches/training_samples_31.dat $SLURM_TMPDIR


# In[1]:


import os
import nibabel as nib
import sys
import tensorflow as tf
import pandas as pd 
import numpy as np
import subprocess
from glob import glob
from tensorflow import keras
from nilearn import image
from sklearn.model_selection import train_test_split


# In[48]:

#sys.path.insert(1, '/home/athurai3/scratch/seeg_contacts_loc')

patches_path = '/home/athurai3/scratch/test_preproc_contact_loc/train_patches/training_samples_31.dat'

#patches_path = os.path.abspath('$SLURM_TMPDIR/training_samples_31.dat')


# In[49]:


#hyperparameters
batch_size = 2 #image data gen to read array generator from c3d*

patch_radius= np.array([31,31,31]) #in voxels -- patch will be shape: 1+2*radius

#3D rotation augmentation; only for training set
num_augment = 2 #number of augmentations per patch
angle_stdev = 30 #stdev of normal distribution for sampling angle (in degrees)

#sampling radius
radius = patch_radius 

num_channels = 2 #ct and corresponding electrode mask


#creating sampling radius argument (R0xR1xR2)
radius_arg = 'x'.join([str(rad) for rad in radius])

#patch shape = 1+2*radius
dims = 1+2*radius


# In[50]:


bps = 4 * num_channels * np.prod(dims)         # Bytes per sample
file_size = os.path.getsize(patches_path) 
num_samples = np.floor_divide(file_size,bps)   # Number of samples
print(file_size)
print(bps)
print(num_samples)

dims = dims.astype('int')
arr_shape_train = (num_samples,dims[0],dims[1],dims[2],num_channels)

arr_train = np.memmap(patches_path,'float32','r',shape=arr_shape_train)
arr_train = np.swapaxes(arr_train,1,3)



# In[25]:


#arr_train = np.swapaxes(arr_train,1,4)
#arr_val = np.swapaxes(arr_val,1,4)



#arr_train_image = arr_train[:,0,:,:,:].reshape(arr_train.shape[0],1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])
#arr_train_label = arr_train[:,1,:,:,:].reshape(arr_train.shape[0],1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])

#arr_val_image = arr_val[:,0,:,:,:].reshape(arr_val.shape[0],1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
#arr_val_label = arr_val[:,1,:,:,:].reshape(arr_val.shape[0],1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])


# In[27]:



# In[51]:


import generator
x_train_1 = arr_train[:,:,:,:,0]
x_train_f = x_train_1.reshape(x_train_1.shape[0], x_train_1.shape[1], x_train_1.shape[2], x_train_1.shape[3], 1)
y_train_1 = arr_train[:,:,:,:,1].astype(int)
y_train_f = y_train_1.reshape(y_train_1.shape[0], y_train_1.shape[1], y_train_1.shape[2], y_train_1.shape[3], 1)

datagen_train = generator.customImageDataGenerator()
new_train = datagen_train.flow(x_train_f, y_train_f, batch_size = 10)


# In[52]:


print(x_train_f.shape)


# In[53]:


print(y_train_f.shape)


# In[54]:

print(np.unique(y_train_f.shape))

#input 
img_shape = (None,None,None,1)
input_layer = keras.layers.Input(img_shape)
x = keras.layers.ZeroPadding3D(padding= ((1,0),(1,0),(1,0)))(input_layer)


#block 1:
x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(16,(3,3,3),padding='same',activation='relu')(x)

out_layer1 = x
x = keras.layers.MaxPooling3D((2,2,2))(x)
x = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)


#block 2:
x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)

out_layer2 = x
x = keras.layers.MaxPooling3D((2,2,2))(x)
#x = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)


#block 3:
x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)

out_layer3 = x
x = keras.layers.MaxPooling3D((2,2,2))(x)
#x = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)(x)


#block 4:
x = keras.layers.Conv3D(256,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(256,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)

out_layer4 = x
x = keras.layers.MaxPooling3D((2,2,2))(x)

#bottleneck
x = keras.layers.Conv3D(512,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(512,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3DTranspose(filters= 128,kernel_size= 2, strides =(2,2,2))(x)
x = keras.layers.Conv3D(256,(2,2,2),padding='same',activation='relu')(x)
x = keras.layers.Concatenate(axis=4)([out_layer4,x])


#expanding path 

#block 5 (opposite 4)
x = keras.layers.Conv3D(256,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(256,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)

x = keras.layers.Conv3DTranspose(64,kernel_size= 2,strides=2,padding='same')(x)
x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Concatenate(axis=4)([out_layer3,x])
x = tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)


#block 6 (opposite 3)
x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(128,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)

x = keras.layers.Conv3DTranspose(32,kernel_size= 2,strides =2,padding='same')(x)
x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Concatenate(axis=4)([out_layer2,x])
#x = tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)

#block 7 (opposite 2)
x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(64,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)

x = keras.layers.Conv3DTranspose(16,kernel_size= 2,strides = 2,padding='same')(x)
x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Concatenate(axis=4)([out_layer1,x])
#x = tf.keras.layers.Dropout(0.1, noise_shape=None, seed=None)(x)


# bloack 8 (opposite 1)
x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)
x = keras.layers.Conv3D(32,(3,3,3),padding='same',activation='relu')(x)
#x = keras.layers.Conv3D(16,(3,3,3),padding='same',activation='relu')(x)


#output layer
x = keras.layers.Cropping3D(cropping=((1, 0), (1, 0), (1, 0)), data_format=None)(x)
x = keras.layers.Conv3D(1,(1,1,1),padding='same',activation='sigmoid')(x)
model = keras.Model(input_layer,x)


# In[55]:


def gen_conv3d_layer(
    filters: int,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
) -> keras.layers.Conv3D:
    return keras.layers.Conv3D(filters, kernel_size, padding="same", activation="relu")


def gen_max_pooling_layer() -> keras.layers.MaxPooling3D:
    return keras.layers.MaxPooling3D((2, 2, 2))


def gen_transpose_layer(filters: int) -> keras.layers.Conv3DTranspose:
    return keras.layers.Conv3DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
    )

def gen_std_block(filters: int, input_):
    x = gen_conv3d_layer(filters)(input_)
    out_layer = gen_conv3d_layer(filters)(x)
    return out_layer, gen_max_pooling_layer()(out_layer)


def gen_opposite_block(filters: int, input_, out_layer):
    x = input_
    for _ in range(3):
        x = gen_conv3d_layer(filters)(x)
    next_filters = filters // 2
    x = gen_transpose_layer(next_filters)(x)
    x = gen_conv3d_layer(next_filters)(x)
    return keras.layers.Concatenate(axis=4)([out_layer, x])


def gen_model() -> keras.Model:
    input_layer = keras.layers.Input((None, None, None, 1))
    x = keras.layers.ZeroPadding3D(padding=((1, 0), (1, 0), (1, 0)))(input_layer)

    out_layer_1, x = gen_std_block(16, x)  # block 1
    out_layer_2, x = gen_std_block(32, x)  # block 2
    out_layer_3, x = gen_std_block(64, x)  # block 3
    out_layer_4, x = gen_std_block(128, x)  # block 4

    # bottleneck
    x = gen_conv3d_layer(256)(x)
    x = gen_conv3d_layer(256)(x)
    x = keras.layers.Conv3DTranspose(filters=128, kernel_size=2, strides=(2, 2, 2))(x)
    x = gen_conv3d_layer(128, (2, 2, 2))(x)
    x = keras.layers.Concatenate(axis=4)([out_layer_4, x])

    x = gen_opposite_block(128, x, out_layer_3)  # block 5 (opposite 4)
    x = gen_opposite_block(64, x, out_layer_2)  # block 6 (opposite 3)
    x = gen_opposite_block(32, x, out_layer_1)  # block 7 (opposite 2)

    # block 8 (opposite 1)
    for _ in range(3):
        x = gen_conv3d_layer(16)(x)

    # output layer
    x = keras.layers.Cropping3D(cropping=((1, 0), (1, 0), (1, 0)), data_format=None)(x)
    x = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation=None)(x)

    return keras.Model(input_layer, x)




























# Compile the model
import math


from keras import backend as K


def dice_metric(y_true, y_pred, episilon = 1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #print(y_true_f)
    #print(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + episilon) / (K.sum(y_true_f) + K.sum(y_pred_f) + episilon)
    return score
    
def dice_loss_3D(y_true, y_pred):
    return (1 - dice_metric(y_true, y_pred))
    

#https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
#mention of negative Dice Loss - mask of 0s/1s, predicted mask is 0-1 bc of sigmoid activation
#avoid this by thresholding the predicted y value
def dice_coeff(y_true, y_pred):
    smooth = 100
    #Flatten - first reshape to flatten ([-1]), then cast to float32 datatype
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), 'float32')
    #threshold to have just values greater than 0.5
    y_pred_f = tf.cast(tf.reshape(y_pred > 0.5, [-1]),'float32')


    intersection = tf.reduce_sum(tf.math.multiply(y_true_f,y_pred_f))
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss_2(y_true, y_pred):
    return(1 - dice_coeff(y_true, y_pred))


optimizer = keras.optimizers.Adam()

loss = ['binary_crossentropy', dice_loss_3D]

metrics = ['binary_accuracy', dice_metric]

model.summary()


model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(new_train, epochs=50, steps_per_epoch = 10)


# In[ ]:


model.save(f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/UNet-keras')


# In[ ]:


import seaborn as sns
#plot loss and metrics
loss_out_path = f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/loss_metrics'
df = pd.DataFrame(history.history)
sns.lineplot(data=df)
pd.DataFrame(history.history).to_csv(loss_out_path)
plt.savefig(f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/lossfunction_biophys.png')

