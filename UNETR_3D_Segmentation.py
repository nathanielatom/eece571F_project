#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/atom/Dropbox/Skule/EECE571F/Project/research-contributions/UNETR/BTCV/networks')


# In[ ]:


import glob
import datetime

from tqdm import tqdm
from tqdm import trange
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
from skimage.transform import resize
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

from unetr import UNETR
# from self_attention_cv.Transformer3Dsegmentation.tranf3Dseg import Transformer3dSeg
# from self_attention_cv.transunet import TransUnet
# from self_attention_cv import TransformerEncoder
# from self_attention_cv import ViT, ResNet50ViT


# In[ ]:


def iou(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    # multiclass IOU: intersection is where classes agree; union is any non-null class
    
    # output shape is (8, 4, 3, 3, 3), one segment for each patch
    outputs = outputs.argmax(axis=1)
    labels = labels.argmax(axis=1)
    
    intersection = (outputs == labels).float().sum((-1, -2, -3)) # Will be zero if Truth=0 or Prediction=0
    union = ((outputs != 0) | (labels != 0)).float().sum((-1, -2, -3)) # Will be zero if both are 0
    
    iou = (intersection + smooth) / (union + smooth) # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean()


# In[ ]:


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets)
        
        intersection = (inputs * targets).sum()                            
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)        
        return 1 - dice


# In[ ]:


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    """
    From https://github.com/ZFTurbo/segmentation_models_3D/blob/cc9f4fdd22387cc1556c77a85c7bea43e541ef1d/segmentation_models_3D/base/functional.py#L259
    as oppose to https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py#L11
    """
    def __init__(self, weight=None, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, pr, gt, smooth=1e-6):
        # clip to prevent NaN's and Inf's
        pr = torch.clamp(pr, smooth, 1 - smooth)
        
        focal_loss = -1 * gt * torch.log(pr) * (self.alpha * (1 - pr) ** self.gamma)
        
        # ce_loss = torch.nn.functional.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        # pt = torch.exp(-ce_loss)
        # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss.mean()


# In[ ]:


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='bool')[y.astype(np.int16)]


# In[ ]:


def generate_brats_batch(prefix, 
                         contrasts, 
                         batch_size=32, 
                         tumour='*', 
                         patient_ids='*',
                         resample_size=(None, None, None),
                         augment_size=None,
                         infinite=True):
    """
    Generate arrays for each batch, for x (data) and y (labels), where the contrast is treated like a colour channel.
    
    Example:
    x_batch shape: (32, 240, 240, 155, 4)
    y_batch shape: (32, 240, 240, 155)
    
    augment_size must be less than or equal to the batch_size, if None will not augment.
    
    """
    file_pattern = '{prefix}/MICCAI_BraTS_2018_Data_Training/{tumour}/{patient_id}/{patient_id}_{contrast}.nii.gz'
    while True:
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
        shape = nib.load(filenames_by_contrast[arbitrary_contrast][0]).get_fdata().shape if resample_size == (None, None, None) else resample_size

        # initialize empty array of batches
        x_batch = np.empty((batch_size, ) + shape + (len(contrasts), )) #, dtype=np.int32)
        y_batch = np.empty((batch_size, ) + shape + (n_classes,)) #, dtype=np.int32)
        num_images = len(filenames_by_contrast[arbitrary_contrast])
        np.random.shuffle(filenames_by_contrast[arbitrary_contrast])
        for bindex in trange(0, num_images, batch_size):
            filenames = filenames_by_contrast[arbitrary_contrast][bindex:bindex + batch_size]
            for findex, filename in enumerate(filenames):
                for cindex, contrast in enumerate(contrasts):

                    # load raw image batches and normalize the pixels
                    tmp_img = nib.load(filename.replace(arbitrary_contrast, contrast)).get_fdata()
                    if resample_size != (None, None, None):
                        tmp_img = resize(tmp_img, resample_size, mode='edge')
                    tmp_img = scalar.fit_transform(tmp_img.reshape(-1, tmp_img.shape[-1])).reshape(tmp_img.shape)
                    x_batch[findex, ..., cindex] = tmp_img

                    # load mask batches and change to categorical
                    tmp_mask = nib.load(filename.replace(arbitrary_contrast, 'seg')).get_fdata()
                    tmp_mask[tmp_mask==4] = 3
                    tmp_mask = to_categorical(tmp_mask, num_classes=4)
                    if resample_size != (None, None, None):
                        tmp_mask = resize(tmp_mask, resample_size, mode='edge')
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
        if not infinite:
            break


# Model Architecture Hyperparameters
# ---

# In[ ]:


prefix = '/home/atom/Documents/datasets/brats' # Adam's Station
output_dir = prefix + '/unet_models/'
batch_size = 6 # 16
contrasts = ['t1ce', 'flair', 't2', 't1']


# In[ ]:


brats_classes = 4
brats_contrasts = 4
brats_x = 80 # 240
brats_y = 80 # 240
brats_z = 80 # 155
resampled_shape = (brats_x, brats_y, brats_z)

patch_size = 16
feature_size = 20
embedding_size = 768 # 1536
msa_heads = 6 # 12
mlp_size = 1024 # 2048

dropout = 0.10
max_epochs = 50
learning_rate = 0.0005


# In[ ]:


class OURUNETR(UNETR):
    
    @property
    def patch_size(self):
        return (patch_size, patch_size, patch_size)
    
    @patch_size.setter
    def patch_size(self, value):
        pass


# In[ ]:


device


# In[ ]:


# model = OURUNETR(
#     in_channels=brats_contrasts,
#     out_channels=brats_classes,
#     img_size=(brats_x, brats_y, brats_z),
#     feature_size=feature_size,
#     hidden_size=embedding_size,
#     mlp_dim=mlp_size,
#     num_heads=msa_heads,
#     pos_embed='perceptron',
#     norm_name='instance',
#     conv_block=True,
#     res_block=True,
#     dropout_rate=dropout)
# model = model.to(device)

from monai.networks.nets import BasicUNet

model = BasicUNet(
            spatial_dims = 3,
            in_channels = brats_contrasts,
            out_channels = brats_classes,
            dropout = 0.10
)
model = model.to(device)




# In[ ]:


# loss = torch.nn.CrossEntropyLoss()
focal_loss = FocalLoss()
# dice_loss = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


model


# Training and Validation
# ---

# In[ ]:


import random, os

brats_dir = '/MICCAI_BraTS_2018_Data_Training/'

data_list_LGG = os.listdir(os.path.join(prefix+brats_dir,'LGG'))
data_list_HGG = os.listdir(os.path.join(prefix+brats_dir,'HGG'))
dataset_file_list = data_list_HGG + data_list_LGG

# shuffle and split the dataset file list
random.seed(42)
file_list_shuffled = dataset_file_list.copy()
random.shuffle(file_list_shuffled)
test_ratio = 0.2

train_file, test_file = file_list_shuffled[0:int(len(file_list_shuffled)*(1-test_ratio))], file_list_shuffled[int(len(file_list_shuffled)*(1-test_ratio)):]

while '.DS_Store' in train_file:
    train_file.remove('.DS_Store')
while '.DS_Store' in test_file:
    test_file.remove('.DS_Store')


# In[ ]:


def ground_truth_segmentation(mask_block):
    """
    For a given block, return the segmentation probability for each patch (reducing over all the voxels in a patch).
    """
    # old method that just picks the segmentation at the centre of the patch:
    # slice near the (slightly off-centre) centre of the patch to choose the class
    # patch_gt = mask_block[..., patch_side // 2::patch_side, patch_side // 2::patch_side, patch_side // 2::patch_side]

    # combine segmentation results (total counts for all classes across patch) then normalize;
    # by summing accross the patch, the segmentation labels can be converted to probabilities to account for 
    # partial volume effects (if half the patch is one segment and half is another, each will get 50% probability)
    # mask (6, 4, 24, 24, 24) -> (6, 4, 3, 8, 3, 8, 3, 8) then sum across the patch dims
    patch_gt = mask_block.reshape(-1, brats_classes, block_side // patch_side, patch_side, block_side // patch_side, patch_side, block_side // patch_side, patch_side).sum(axis=-1).sum(axis=-2).sum(axis=-3)
    return patch_gt / patch_side ** 3


# In[ ]:


def train():
    model.train()
    running_loss = 0
    count = 0
    for img, mask in generate_brats_batch(prefix, contrasts, batch_size=batch_size, patient_ids=train_file, infinite=False, resample_size=resampled_shape):
        # img (8, 240, 240, 155, 4) -> (8, 4, 240, 240, 155)
        img, mask = np.rollaxis(img, -1, 1), np.rollaxis(mask, -1, 1)
        img_gpu, mask_gpu = torch.FloatTensor(img).to(device), torch.FloatTensor(mask).to(device)
        
        optimizer.zero_grad()
        output = model(img_gpu)
        predictions = torch.softmax(output, axis=1)
        # output shape is (8, 4, 240, 240, 155), one segment for each voxel

        # current_loss = loss(output, patch_gt.argmax(axis=1))
        current_loss = focal_loss(predictions, mask_gpu) # + dice_loss(predictions, patch_gt)
                    
        current_loss.backward()
        optimizer.step()
        running_loss += current_loss.item()
        count += 1
        print(f'Training batch loss: {running_loss / count}')
        writer.add_scalar('Training batch loss', running_loss / count)
        writer.flush()
    return running_loss / count


# In[ ]:


def validate():      
    model.eval()
    with torch.no_grad():
        running_loss = 0
        running_iou = 0
        count = 0
        for img, mask in generate_brats_batch(prefix, contrasts, batch_size=batch_size, patient_ids=test_file, infinite=False, resample_size=resampled_shape):
            # img (8, 240, 240, 155, 4) -> (8, 4, 240, 240, 155)
            img, mask = np.rollaxis(img, -1, 1), np.rollaxis(mask, -1, 1)
            img_gpu, mask_gpu = torch.FloatTensor(img).to(device), torch.FloatTensor(mask).to(device)
            
            output = model(img_gpu)
            predictions = torch.softmax(output, axis=1)
                        
            # current_loss = loss(output, patch_gt.argmax(axis=1))
            current_loss = focal_loss(predictions, mask_gpu) # + dice_loss(predictions, patch_gt)
                        
            running_iou += iou(predictions, mask_gpu)
            running_loss += current_loss.item()
            count += 1
    return running_loss / count, running_iou / count


# In[ ]:


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/unetr/trainer_{}'.format(timestamp))
best_val_loss = np.inf

for epoch in range(max_epochs):
    train_loss = train()
    val_loss, val_iou = validate()
    
    print(f'Epoch {epoch}: LOSS train {train_loss}; validation {val_loss}; validation IOU {val_iou}')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training' : train_loss, 'Validation' : val_loss, 'IOU': val_iou}, epoch)
    writer.flush()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = output_dir + f'model_{timestamp}_{epoch}'
        torch.save(model.state_dict(), model_path)


# In[ ]:





# In[ ]:




