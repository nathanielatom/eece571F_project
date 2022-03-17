import os
import glob

import tqdm
import random
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as sci
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
print(devices)
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
# from keras_preprocessing.image import ImageDataGenerator

scalar = MinMaxScaler()
def generate_brats_batch(file_pattern, contrasts, batch_size=2, tumour='*', patient_ids='*', augment_size=None):
    """
    Generate arrays for each batch, for x (data) and y (labels), where the contrast is treated like a colour channel.
    
    Example:
    x_batch shape: (32, 240, 240, 155, 4)
    y_batch shape: (32, 240, 240, 155)
    
    augment_size must be less than or equal to the batch_size, if None will not augment.
    
    """
    n_classes = 4

    # get list of filenames for every contrast available
    keys = dict(prefix=prefix, tumour=tumour)
    filenames_by_contrast = {}
    for contrast in contrasts:
        filenames_by_contrast[contrast] = glob.glob(file_pattern.format(contrast=contrast, patient_id=patient_ids, **keys)) if patient_ids == '*' else []
        if patient_ids != '*':
            contrast_files = []
            for patient_id in patient_ids:
                contrast_files.extend(glob.glob(file_pattern.format(contrast=contrast, patient_id=patient_id, **keys)))
            filenames_by_contrast[contrast] = contrast_files
    
    # get the shape of one 3D volume and initialize the batch lists
    arbitrary_contrast = contrasts[0]
    shape = nib.load(filenames_by_contrast[arbitrary_contrast][0]).get_fdata().shape
    x_batch = np.empty((batch_size, ) + shape + (len(contrasts), )) #, dtype=np.int32)
    y_batch = np.empty((batch_size, ) + shape + (n_classes,)) #, dtype=np.int32)
    num_images = len(filenames_by_contrast[arbitrary_contrast])
    print(num_images)
    np.random.shuffle(filenames_by_contrast[arbitrary_contrast])
    for bindex in tqdm.tqdm_notebook(range(0, num_images, batch_size), total=num_images):
        filenames = filenames_by_contrast[arbitrary_contrast][bindex:bindex + batch_size]
        for findex, filename in enumerate(filenames):
            for cindex, contrast in enumerate(contrasts):

                # load raw image batches and normalize the pixels
                tmp_img = nib.load(filename.replace(arbitrary_contrast, contrast)).get_fdata()
                tmp_img = scalar.fit_transform(tmp_img.reshape(-1, tmp_img.shape[-1])).reshape(tmp_img.shape)
                x_batch[findex, ..., cindex] = tmp_img

                # load mask batches and change to categorical
                tmp_mask = nib.load(filename.replace(arbitrary_contrast, 'seg')).get_fdata()
                tmp_mask[tmp_mask==4] = 3
                tmp_mask = to_categorical(tmp_mask, num_classes = 4)
                y_batch[findex] = tmp_mask
        
        if bindex + batch_size > num_images:
            x_batch, y_batch = x_batch[:num_images - bindex], y_batch[:num_images - bindex]
        if augment_size is not None:
            # x_aug, y_aug = augment(x_batch, y_batch, augment_size)
            x_aug = None
            y_aug = None
            yield np.append(x_batch, x_aug), np.append(y_batch, y_aug)
        else:
            yield x_batch, y_batch


tumours = ['LGG','HGG']
# prefix = '/Users/jasonfung/Documents/EECE571' # Jason's Macbook
prefix = 'C:/Users/Fungj/Documents/EECE_571F' # Jason's Desktop
brats_dir = '/MICCAI_BraTS_2018_Data_Training/'
# prefix = '/home/atom/Documents/datasets/brats' # Adam's Station
file_pattern = '{prefix}/MICCAI_BraTS_2018_Data_Training/{tumour}/{patient_id}/{patient_id}_{contrast}.nii.gz'
# patient_id = 'Brats18_TCIA09_620_1'
contrasts = ['t1ce', 'flair', 't2']
tumours = ['LGG', 'HGG']

data_list_LGG = os.listdir(os.path.join(prefix+brats_dir,tumours[0]))
data_list_HGG = os.listdir(os.path.join(prefix+brats_dir,tumours[1]))
dataset_file_list = data_list_HGG + data_list_LGG

# shuffle and split the dataset file list
import random
random.seed(42)
file_list_shuffled = dataset_file_list.copy()
random.shuffle(file_list_shuffled)
test_ratio = 0.2

train_file, test_file = file_list_shuffled[0:int(len(file_list_shuffled)*(1-test_ratio))], file_list_shuffled[int(len(file_list_shuffled)*(1-test_ratio)):]

train_datagen = generate_brats_batch(file_pattern, contrasts, batch_size = 2, patient_ids = train_file) # first iteration
test_datagen = generate_brats_batch(file_pattern, contrasts, batch_size = 2,patient_ids = test_file) # first iteration


import segmentation_models_3D as sm 
sm.set_framework('tf.keras')

# data parameters
x_size = None
y_size = None
z_size = None
contrast_channels = 3
input_shape = (x_size, y_size, z_size, contrast_channels)
n_classes = 4

# define Hyper Parameters
LR = 0.0001
activation = 'softmax'

encoder_weights = 'imagenet'
BACKBONE = 'resnet50'
optim = tf.keras.optimizers.Adam(LR)
class_weights = [0.25, 0.25, 0.25, 0.25]

# limit memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Define Loss Functions
dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1*focal_loss)
metrics = [sm.metrics.IOUScore(threshold = 0.5), sm.metrics.FScore(threshold = 0.5)]

# Define the model being used. In this case, UNet
model = sm.Unet(backbone_name= BACKBONE,
                classes = n_classes,
                input_shape = input_shape,
                encoder_weights = encoder_weights,
                activation = activation,
                decoder_block_type = 'transpose')

model.compile(optimizer = optim, loss = total_loss, metrics = metrics)

with tf.device('/device:CPU:0'):
	history = model.fit(train_datagen,
	                    epochs = 50,
	                    verbose = 1,
	                    validation_data = test_datagen)