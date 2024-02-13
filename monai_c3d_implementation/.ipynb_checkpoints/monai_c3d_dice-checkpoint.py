#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
import tempfile
import monai
from monai.utils import first, set_determinism, progress_bar
from monai.data import PILReader
from monai.transforms import (LoadImage, LoadImaged,ScaleIntensityRangePercentiles, Resized, Compose, SaveImage, 
ScaleIntensity,RandScaleIntensity, SpatialPadd)
from monai.config import print_config
from monai.data import CacheDataset, NiftiSaver
import glob
import torch
import nibabel as nib
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandAffined,
    SpatialPadd,
    Rand3DElasticd,
    RandFlipd,
    ResizeWithPadOrCropd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PatchDataset
from monai.config import print_config
from monai.apps import download_and_extract
import matplotlib.pyplot as plt


# In[2]:


#hyperparameters
batch_size = 2 #image data gen to read array generator from c3d*

patch_radius= np.array([15,15,15]) #in voxels -- patch will be shape: 1+2*radius

#3D rotation augmentation; only for training set
num_augment = 2 #number of augmentations per patch
angle_stdev = 30 #stdev of normal distribution for sampling angle (in degrees)

#sampling radius
radius = patch_radius 

num_channels = 2 #ct and contacts


#creating sampling radius argument (R0xR1xR2)
radius_arg = 'x'.join([str(rad) for rad in radius])

#patch shape = 1+2*radius
dims = 1+2*radius



# In[3]:


fname = '/home/athurai3/projects/ctb-akhanf/athurai3/unet_contact_seg/sample_data/sub-P001/sub-P001_desc-znorm_patches.dat'
bps = 4 * num_channels * np.prod(dims)         # Bytes per sample
file_size = os.path.getsize(fname) 
num_samples = np.floor_divide(file_size,bps)   # Number of samples
print(file_size)
print(bps)
print(num_samples)


#can change first index from num_samples to try with training on smaller number of jobs, kjupyter job
dims = dims.astype('int')
arr_shape_train = (num_samples, dims[0],dims[1],dims[2],num_channels)

arr_train = np.memmap(fname,'float32','r',shape=arr_shape_train)
#arr_train = np.swapaxes(arr_train,1,3)
print(arr_shape_train)


# In[5]:


fname_val = '/home/athurai3/projects/ctb-akhanf/athurai3/unet_contact_seg/sample_data/sub-P002/sub-P002_desc-znorm_patches.dat'

bps = 4 * num_channels * np.prod(dims)         # Bytes per sample
file_size_val = os.path.getsize(fname_val) 
num_samples_val = np.floor_divide(file_size_val,bps)   # Number of samples
print(file_size_val)
print(bps)
print(num_samples_val)


#can change first index with num_samples_val to try with training on smaller number of jobs, kjupyter job
dims = dims.astype('int')
arr_shape_val = (num_samples_val,dims[0],dims[1],dims[2],num_channels)

arr_val = np.memmap(fname_val,'float32','r',shape=arr_shape_val)
#arr_val = np.swapaxes(arr_val,1,3)
print(arr_shape_val)


# In[6]:


arr_train = np.swapaxes(arr_train,1,4) #swap second axis with last -  that of channels (image/label)
arr_val = np.swapaxes(arr_val, 1, 4) #swap second axis with last - that of channels (image/label)

arr_train_image = arr_train[:,0,:,:,:].reshape(arr_train.shape[0],1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])
arr_train_label = arr_train[:,1,:,:,:].reshape(arr_train.shape[0],1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])

arr_val_image = arr_val[:,0,:,:,:].reshape(arr_val.shape[0],1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
arr_val_label = arr_val[:,1,:,:,:].reshape(arr_val.shape[0],1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])

thresh_train_label = np.where(arr_train_label[:][:] > 0, 1, arr_train_label) #binarize labels
thresh_val_label = np.where(arr_val_label[:][:] > 0, 1, arr_val_label) #binarize labels


# In[7]:


arr_train_dict= [{"image": ct, "label": contact} for ct, contact in zip(arr_train_image,thresh_train_label)]
arr_val_dict= [{"image": ct, "label": contact} for ct, contact in zip(arr_val_image,thresh_val_label)]


# In[8]:


train_transforms = Compose(
    [       
        SpatialPadd(
            keys = ("image", "label"),
            spatial_size=(32,32,32)
        ),
        EnsureTyped (keys= ("image", "label"))
    ]
)

val_transforms = Compose(
    [
        SpatialPadd(
            keys = ("image", "label"),
            spatial_size=(32,32,32)
        ),
        EnsureTyped (keys= ("image", "label"))
    ]
)

train_patches_dataset = CacheDataset(data=arr_train_dict ,transform = train_transforms, cache_rate =0.75, copy_cache=False, progress=True) # dataset with cache mechanism that can load data and cache deterministic transforms’ result during training.
validate_patches_dataset = CacheDataset(data=arr_val_dict, transform = val_transforms, cache_rate = 0.75, copy_cache=False,progress=True)


# In[11]:


batch_size = 32
training_steps = int(num_samples / batch_size) # number of training steps per epoch
validation_steps = int(num_samples_val/ batch_size) # number of validation steps per epoch


# In[12]:


train_loader = DataLoader(train_patches_dataset, batch_size=batch_size, shuffle=False, num_workers=2) #num_workers is number of cpus we request
val_loader = DataLoader(validate_patches_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# In[13]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout = 0.4,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceCELoss(sigmoid = True) #revisit
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
dice_metric = DiceMetric(include_background=True, reduction="mean")


# In[16]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# In[ ]:


import time
#from pytorchtools import EarlyStopping

date = '2024Feb5'
start = time.time() # initializing variable to calculate training time


from monai.transforms import Activations, Activationsd, AsDiscrete, AsDiscreted, EnsureTyped, SaveImaged


val_interval = 2
best_metric = -1
best_metric_epoch = -1
max_epochs = 750
epoch_loss_values = [0]
val_dice_metric_values = [0]
val_loss_values = [0]
post_transforms = Compose(
    [
        Activations(sigmoid=True),
        AsDiscrete(threshold = 0.5),
        #SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=monai_test_dir, resample=False),
    ]
)


for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    progress_bar(
        index = epoch+1, 
        count=max_epochs, 
        desc = f"epoch {epoch+1}, training loss: {epoch_loss_values[-1]:.4f}, validation metric: {val_dice_metric_values[-1]:.4f}, validation loss: {val_loss_values[-1]:.4f}",
        newline = True)
    step = 0
    for batch_data in train_loader:
        step +=1
        images = batch_data["image"].cuda()
        labels = batch_data["label"].cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_patches_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    
    if (epoch + 1) % val_interval == 0:
        model.eval()
    
        with torch.no_grad():
            avg_val_loss = 0
            val_images = None
            val_labels = None
            val_outputs = None
            for batch_valdata in val_loader:
                val_images, val_labels = (batch_valdata["image"].cuda(), batch_valdata["label"].cuda())
                #sliding window size and batch size for inference
                roi_size = (32, 32, 32)
                sw_batch_size = 32
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                #val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                #batch_valdata['label'] = val_labels
            
                #val_outputs = post_transforms(i) for i in decollate_batch(val_outputs)]
                val_outputs = post_transforms(val_outputs)
                print(torch.unique(val_outputs))
                dice_metric(y_pred = val_outputs, y = val_labels)
                #metric_count += len(value)
                #metric_sum += value.item() * len(value)
                
                val_loss = loss_function(val_outputs, val_labels)
                avg_val_loss += val_loss.item()
            
            avg_val_loss = avg_val_loss.item()/validation_steps
            val_loss_values.append(avg_val_loss)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            val_dice_metric_values.append(metric)
            
            print(f'Validation Metric: {metric}')
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/UNET/{date}_diceloss_checkpoint.pt')
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

end = time.time()
time = end - start
print(time)


# In[ ]:


with open (f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/UNET/{date}_dicelossmodel_stats.txt', 'w') as file:  
    file.write(f'training time: {time}\n')  
    file.write(f'training loss: {epoch_loss_values[:]}')#\n validation loss: {early_stopping.val_loss_min}\n')
    file.write(f'validation dice metric: {val_dice_metric_values[:]}')


# In[ ]:


plt.figure(figsize=(14,8))
plt.plot(list(range(len(epoch_loss_values))), epoch_loss_values, label="Training Loss")
#plt.plot(list(range(len(val_loss_values))), val_loss_values , label="Validation Loss")
plt.grid(True, "both", "both")
plt.legend()
plt.savefig(f'/home/athurai3/scratch/seeg_contacts_loc/derivatives/UNET/{date}_dicelossfunction.png')