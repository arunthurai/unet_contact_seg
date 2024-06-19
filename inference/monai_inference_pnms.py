import os
#from monai.data import PILReader
from monai.transforms import (LoadImage, LoadImaged,ScaleIntensityRangePercentiles, Resized, Compose, SaveImage, 
ScaleIntensity,RandScaleIntensity, SpatialPadd)
#from monai.config import print_config
#from monai.data import CacheDataset#, NiftiSaver
import glob
import torch
import nibabel as nib
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ProbNMS
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PatchDataset
#from monai.apps import download_and_extract
from monai.inferers import sliding_window_inference
#import matplotlib.pyplot as plt
import glob

import pandas as pd
import numpy as np

model_desc = 'Jun17_patch95_4layers_diceCE'
model_fname = f'/scratch/athurai3/monai_outputs/UNET/{model_desc}/checkpoint.pt'

def df_to_fcsv(input_df, output_fcsv):
    with open(output_fcsv, 'w') as fid:
        fid.write("# Markups fiducial file version = 4.11\n")
        fid.write("# CoordinateSystem = 0\n")
        fid.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
    
    out_df={'node_id':[],'x':[],'y':[],'z':[],'ow':[],'ox':[],'oy':[],'oz':[],
        'vis':[],'sel':[],'lock':[],'label':[],'description':[],'associatedNodeID':[]
    }
    
    for idx,ifid in input_df.iterrows():
        out_df['node_id'].append(idx+1)
        out_df['x'].append(ifid.iloc[0])
        out_df['y'].append(ifid.iloc[1])
        out_df['z'].append(ifid.iloc[2])
        out_df['ow'].append(0)
        out_df['ox'].append(0)
        out_df['oy'].append(0)
        out_df['oz'].append(0)
        out_df['vis'].append(1)
        out_df['sel'].append(1)
        out_df['lock'].append(1)
        out_df['label'].append(str(idx+1)) #no label for points for now
        out_df['description'].append('')
        out_df['associatedNodeID'].append('')

    out_df=pd.DataFrame(out_df)
    out_df.to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False, float_format = '%.3f')

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

test_dir = '/scratch/athurai3/val_thesis'
orig_dir = '/scratch/athurai3/preproc_final'

subjects = [identifier for identifier in os.listdir(test_dir) if "sub-" in identifier]
subjects = sorted(subjects)
# subjects = ['sub-P020']  #identify which specific subjects to run model on

test_ct = []
test_subjects = []
for subject in subjects:
    subject_ct = f'{orig_dir}/{subject}/{subject}_res-0p4mm_ct.nii.gz'
    if glob.glob(subject_ct):
        print(subject)
        test_subjects.append(subject)
        test_ct.append(subject_ct)

test_data = [{"image": image} for image in test_ct]

test_transforms = Compose( #loading full image
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"])])

test_ds = Dataset(data = test_data, transform = test_transforms)

test_loader = DataLoader(test_ds, batch_size = 1, num_workers = 1)


post_transforms = Compose(
    [   
        Activations(sigmoid = True),
        #AsDiscrete(threshold = 0.5)
    ]
)

probNMS_transform = Compose(
     [
        ProbNMS(spatial_dims = 3, box_size = [10,10,10])
     ]
)

pred_imgs = []
model.load_state_dict(torch.load(model_fname, 
                                 map_location=torch.device(device)))
model.eval()

with torch.no_grad():
    for test_img in test_loader:
        test_inputs = test_img['image'].to('cpu')
        roi_size = (96, 96, 96) #change to reflect patch size model was trained on
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
        pred_img = probNMS_transform(pred_img[0][0]) #to get coordinates
        pred_imgs.append(pred_img)
        #pred_img = post_transforms(pred_img) #if wanting the image
        #pred_imgs.append(pred_img.cpu().numpy())
        del pred_img

if not os.path.exists(f'/scratch/athurai3/monai_outputs/{model_desc}/prob_nms'):
    os.makedirs(f'/scratch/athurai3/monai_outputs/{model_desc}/prob_nms')

if not os.path.exists(f'/scratch/athurai3/monai_outputs/{model_desc}'):
    os.makedirs(f'/scratch/athurai3/monai_outputs/{model_desc}')


#creating fcsv dataframes from prob_nms outputs to view in slicer
patch_size = 'patch96'

for sub in range(len(pred_imgs)):
    pred_test_pnms = pred_imgs[sub]
    print(test_ct[sub])

    #print(pred_test_pnms.shape)
    file_name_pred = f'/scratch/athurai3/monai_outputs/{model_desc}/prob_nms/{test_subjects[sub]}_desc-{patch_size}_unet_pnms.fcsv'
    orig_test = nib.load(test_ct[sub])
    print(f'creating {test_subjects[sub]} prediction')

    M = orig_test.affine[:3,:3]
    abc = orig_test.affine[:3,3]

    pred_test_pnms_df = pd.DataFrame(pred_test_pnms, columns=['probability', 'x', 'y', 'z'])

    coords = pred_test_pnms_df[['x', 'y', 'z']].to_numpy()

    #print(coords)

    #create empty array to fill with transformed coordinates
    transformed_coords = np.zeros(coords.shape)
    print('Transforming from voxel to coordinate space')
    #apply tranformations row by row
    for i in range(len(transformed_coords)):
        vec = coords[i,:]
        tvec = M.dot(vec) + abc
        transformed_coords[i,:] = tvec[:3]

    ras_points = pd.DataFrame(transformed_coords, columns = ['x','y','z'])

    print(ras_points)
        
    df_to_fcsv(ras_points, file_name_pred)

#creating nifti images

# for i in range(len(pred_imgs)):
#     prediction = pred_imgs[i][0][0]
#     file_name_pred = f'/scratch/athurai3/monai_outputs/{model_desc}/{test_subjects[i]}_res-0p4mm_desc-pred_ct.nii.gz'
#     orig_test = nib.load(test_ct[i])
#     print(f'creating {subjects[i]} prediction')
#     file = nib.Nifti1Image(prediction, affine = orig_test.affine, header = orig_test.header)
#     nib.save(file, file_name_pred)
#     print(f'{subjects[i]} complete')

end = time.time()

time_pred = end-start
print('Time for Prediction', time_pred)
print('Avg Time Per Subject', time_pred/len(subjects))