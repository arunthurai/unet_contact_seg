#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import csv
import shutil
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
from statistics import NormalDist
#from pytablewriter import MarkdownTableWriter,ExcelXlsxTableWriter,CsvTableWriter
import nibabel as nib
import json
from skimage import morphology
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table
import matplotlib as plt
import matplotlib.pyplot as plt
import os
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
		out_df['label'].append(str(ifid.iloc[3]))
		out_df['description'].append('')
		out_df['associatedNodeID'].append('')

	out_df=pd.DataFrame(out_df)
	out_df.to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False, float_format = '%.3f')

# Function to calculate the shortest distance from a point to a line
def line_point_distance(point, line_points):
    p0 = np.array(point)
    p1, p2 = np.array(line_points)
    d = np.cross(p2-p1, p1-p0) / np.linalg.norm(p2-p1)
    return np.linalg.norm(d)

def inside_elec_mask(point, mask_data, mask_affine):
    voxel_coords = np.linalg.inv(mask_affine).dot(np.append(point, 1))[:3]
    voxel_coords = np.round(voxel_coords).astype(int)

    # Check bounds and if the voxel is within the brain mask
    if (0 <= voxel_coords[0] < mask_data.shape[0] and
        0 <= voxel_coords[1] < mask_data.shape[1] and
        0 <= voxel_coords[2] < mask_data.shape[2]):
        return mask_data[tuple(voxel_coords)] > 0
    else:
        return False
    
def filter_points(points_df, mask_data, mask_affine):
    filtered_points = []
    for _, row in points_df.iterrows():
        point = row[[1, 2, 3]].to_numpy()
        if inside_elec_mask(point, mask_data, mask_affine):
            filtered_points.append(row)
    return pd.DataFrame(filtered_points)
    

model_desc = 'Mar16_patch95_4layers_diceCE'
pnms_dir = f'/scratch/athurai3/monai_outputs/{model_desc}/prob_nms'
gt_dir = '/project/6050199/athurai3/seeg_data_final'
old_dir = '/project/6050199/cfmm-bids/Khan/clinical_imaging/epi_ieeg/atlasreg'


subjects = []

for root, dirs, files in os.walk(pnms_dir):
    for file in files:
        subject = file.split('_')[0]
        if subject not in subjects:
            print(subject)
            subjects.append(subject)

print(subjects)

test_seega = []
test_electrode_mask = []
test_fcsv = []
test_transforms = []

     
for sub in subjects.sort():
    print(sub)
    final_fname_t1 = f'{pnms_dir}/{sub}_space-T1w_desc-unet_pnms.fcsv'
    orig_pnms = pd.read_csv(f'{pnms_dir}/{sub}_prob_nms.fcsv', skiprows=3, header = None)
    ct_t1_trans = np.loadtxt(f'{gt_dir}/{sub}/{sub}_desc-rigid_from-ct_to-T1w_type-ras_ses-post_xfm.txt')
    electrode_mask = f'/scratch/athurai3/preproc_final/{sub}/{sub}_res-0p4mm_desc-electrode_mask.nii.gz'

    elec_img_data, elec_affine = nib.load(electrode_mask).get_fdata(), nib.load(electrode_mask).affine

    print('Filtering by Mask')

    filtered = filter_points(orig_pnms, elec_img_data, elec_affine)

    trans = ct_t1_trans

    M = trans[:3,:3]
    abc = trans[:3,3]

    #create empty array to fill with transformed coordinates
    t1_coords = np.zeros(filtered[[1,2,3]].shape)
    #print(t1_coords)
    ct_coords = filtered[[1,2,3]].to_numpy()
    #print(ct_coords)
    print('Transforming to T1 Space')
    #apply tranformations row by row
    for i in range(len(t1_coords)):
        vec = ct_coords[i,:]
        tvec = M.dot(vec) + abc
        t1_coords[i,:] = tvec[:3]

    t1_points = pd.DataFrame(t1_coords, columns = ['x','y','z'])
    t1_points['label'] = orig_pnms[11].reset_index(drop = True)

    df_to_fcsv(t1_points, final_fname_t1)

print('Loop 2: Labelling')  
for subject in subjects:
    print(f'{subject}') 
    
    final_fname_t1 = f'{pnms_dir}/{subject}_space-T1w_desc-unet_pnms.fcsv'
    
    if glob.glob(final_fname_t1):
    
        df = pd.read_csv(final_fname_t1, header = 2)
        contacts = df[["x","y","z"]].to_numpy()
        df_te = pd.read_csv(f'{gt_dir}/{subject}/{subject}_actual.fcsv', header = 2)
        te_coords = df_te[['x','y','z','label']]


        # Group by label and calculate the defining points for each line
        line_definitions = te_coords.groupby('label').apply(lambda g: g[['x', 'y', 'z']].values).to_dict()



        # Determine the closest line for each point and assign labels
        point_labels = [None] * len(contacts)
        for i, point in enumerate(contacts):
            min_distance = np.inf
            closest_label = None
            for label, line_points in line_definitions.items():
                distance = line_point_distance(point, line_points)
                if distance < min_distance:
                    min_distance = distance
                    closest_label = label
            point_labels[i] = closest_label

        list_of_distances_from_target = []
        for its_label, apoint in enumerate(contacts):
          distance_from_target_point = np.sqrt(np.sum((apoint - line_definitions[point_labels[its_label]][0])**2))
          list_of_distances_from_target.append(distance_from_target_point)

        contacts_with_labels = df[['x', 'y', 'z']]
        contacts_with_labels['label']= point_labels
        contacts_with_labels['distance']= list_of_distances_from_target

        df_sorted = contacts_with_labels.sort_values(by=['label', 'distance'])

        # Create a new column that indicates the ordering within each group
        df_sorted['order_within_label'] = df_sorted.groupby('label').cumcount() + 1
        # Concatenate 'label' and 'order_within_label' to create a new feature
        df_sorted['label_order'] = df_sorted['label'] + df_sorted['order_within_label'].astype(str)

        labeled_contacts = df_sorted[["x","y","z",'label_order']]

        df_to_fcsv(labeled_contacts, final_fname_t1)

