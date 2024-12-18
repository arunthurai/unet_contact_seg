import os
import shutil
from monai.utils import first, set_determinism, progress_bar
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
import tempfile
import monai
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
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    NormalizeIntensityd,
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
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
import glob
import argparse


model_desc = 'Apr9_sdt_patch95_4layers'
model_fname = f'/scratch/athurai3/monai_outputs/UNET/{model_desc}/checkpoint.pt'

# parser = argparse.ArgumentParser(description='PyTorch Example')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# args = parser.parse_args()
# args.device = None
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')

import time

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    dropout = 0.2,
    norm=Norm.BATCH,
).to(device)

test_dir = '/scratch/athurai3/val_0p4mm'
orig_dir = '/scratch/athurai3/preproc_0p4mm'
subjects = [identifier for identifier in os.listdir(test_dir) if "sub-" in identifier]
subjects = sorted(subjects)

test_ct = []
for subject in subjects:
    subject_ct = f'{orig_dir}/{subject}/{subject}_res-0p4mm_desc-z_norm_ct.nii.gz'
    if glob.glob(subject_ct):
        print(subject)
        test_ct.append(subject_ct)

test_data = [{"image": image} for image in test_ct]

test_transforms = Compose( #loading full image
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"])])

test_ds = Dataset(data = test_data, transform = test_transforms)

test_loader = DataLoader(test_ds, batch_size = 1, num_workers = 1)


post_transforms = post_transforms = Compose(
    [
        Activations(sigmoid = True),
        #AsDiscrete(threshold = 0.5)
    ]
)


pred_imgs = []
model.load_state_dict(torch.load(model_fname, 
                                 map_location=torch.device(device)))
model.eval()

with torch.no_grad():
    for test_img in test_loader:
        test_inputs = test_img['image'].to('cpu')
        roi_size = (96, 96, 96)
        sw_batch_size = 8
        pred_img = sliding_window_inference(inputs = test_inputs, 
                                            roi_size=roi_size, 
                                            sw_batch_size = sw_batch_size, 
                                            predictor = model.to(device),
                                            overlap = 0.15, 
                                            mode = 'constant', 
                                            sw_device = device, 
                                            device = device, 
                                            progress=True)
        pred_imgs.append(pred_img.cpu().numpy())
        del pred_img


end = time.time()

time_pred = end-start
print('Time for Prediction', time_pred)
print('Avg Time Per Subject', time_pred/len(subjects))

pred_imgs = post_transforms(pred_imgs)


if not os.path.exists(f'/scratch/athurai3/monai_outputs/{model_desc}'):
    os.makedirs(f'/scratch/athurai3/monai_outputs/{model_desc}')

for i in range(len(pred_imgs)):
    prediction = pred_imgs[i][0][0].cpu()
    file_name_pred = f'/scratch/athurai3/monai_outputs/{model_desc}/{subjects[i]}/{subjects[i]}_res-0p4mm_desc-sdt_pred_ct.nii.gz'
    orig_test = nib.load(test_ct[i])
    print(f'creating {subjects[i]} prediction')
    file = nib.Nifti1Image(prediction.detach().numpy(), affine = orig_test.affine, header = orig_test.header)
    nib.save(file, file_name_pred)
    print(f'{subjects[i]} complete')