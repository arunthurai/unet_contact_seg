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
from mpl_toolkits.mplot3d import axes3d, Axes3D

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
    

model_desc = 'May16_patch95_4layers_diceCE'
test_dir = '/scratch/athurai3/val_final'
gt_dir = '/project/6050199/athurai3/seeg_data_final'
preproc_dir = '/scratch/athurai3/preproc_final'
output_dir = f'/scratch/athurai3/monai_outputs/{model_desc}/coordinates'

if not os.path.exists(f'{output_dir}'):
    os.makedirs(f'{output_dir}')

subjects = [identifier for identifier in os.listdir(test_dir) if "sub-" in identifier]
subjects.sort()


for subject in subjects:
    print(subject)
    subj_pred = f'/scratch/athurai3/monai_outputs/{model_desc}/{subject}_res-0p4mm_desc-pred_ct.nii.gz'
    subj_electrode_mask = f'{preproc_dir}/{subject}/{subject}_res-0p4mm_desc-electrode_mask.nii.gz'
  
    final_fname_ct = f'{output_dir}/{subject}_res-0p4mm_desc-space-ct_unet.fcsv'
    final_fname_t1 = f'{output_dir}/{subject}_res-0p4mm_desc-space-T1w_unet.fcsv'
    subj_transform = f'{gt_dir}/{subject}/{subject}_desc-rigid_from-ct_to-T1w_type-ras_ses-post_xfm.txt'

    print(subj_pred)
    print(subj_transform)
    print(subj_electrode_mask)
    if glob.glob(final_fname_ct) and glob.glob(final_fname_t1):
        print(f'{subject} completed!')

    else:
        print('blah')

        pred_img = nib.load(subj_pred)

        electrode_img_data = nib.load(subj_electrode_mask).get_fdata()

        pred_data = np.where(pred_img.get_fdata()>0.5, 1.0, 0.0)

        print('Multpliying by electrode mask')
        labelled_pred = label(pred_data*electrode_img_data)

        pointlist_pred={
            'x':[],
            'y':[],
            'z':[],
            'labels':[],
            'area': []
        }

        print('Finding Centroids of Segmented Contacts')
        for regions in regionprops(labelled_pred):
            pointlist_pred['x'].append(regions.centroid[0])
            pointlist_pred['y'].append(regions.centroid[1])
            pointlist_pred['z'].append(regions.centroid[2])
            pointlist_pred['labels'].append(regions.label)
            #pointlist_pred['area'].append(regions.area)


        df_pred_sub = pd.DataFrame(pointlist_pred)

        # voxel_volume = np.abs(np.prod(np.diag(pred_img.affine)[:3]))

        # df_pred_sub['area_mm3'] = df_pred_sub['area'] * voxel_volume

        # area_thresh = [voxel_volume, 15.0]

        # df_pred_sub['size_ok'] = True

        # df_pred_sub['area_mm3'] = df_pred_sub['area'] * voxel_volume
        # df_pred_sub.loc[df_pred_sub['area_mm3'] < area_thresh[0],
        #                 'size_ok'] = False
        # df_pred_sub.loc[df_pred_sub['area_mm3'] > area_thresh[1],
        #                 'size_ok'] = False

        # size_filter_pred = df_pred_sub.loc[df_pred_sub['size_ok']==True]

        coords = df_pred_sub[['x', 'y', 'z']].to_numpy()

        #take zooms/translations from image affine
        M = pred_img.affine[:3,:3]
        abc = pred_img.affine[:3,3]

        #create empty array to fill with transformed coordinates
        transformed_coords = np.zeros(coords.shape)
        print('Transforming from voxel to coordinate space')
        #apply tranformations row by row
        for i in range(len(transformed_coords)):
            vec = coords[i,:]
            tvec = M.dot(vec) + abc
            transformed_coords[i,:] = tvec[:3]

        #create new df of transformed points
        ras_points = pd.DataFrame(transformed_coords, columns = ['x','y','z'])
        ras_points['label'] = df_pred_sub['labels'].reset_index(drop = True)
        print('Saving File')
        df_to_fcsv(ras_points, final_fname_ct)

        trans = np.loadtxt(subj_transform)

        M = trans[:3,:3]
        abc = trans[:3,3]

        #create empty array to fill with transformed coordinates
        t1_coords = np.zeros(ras_points[['x','y','z']].shape)
        ct_coords = ras_points[['x','y','z']].to_numpy()
        print('Transforming to T1 Space')
        #apply tranformations row by row
        for i in range(len(t1_coords)):
            vec = ct_coords[i,:]
            tvec = M.dot(vec) + abc
            t1_coords[i,:] = tvec[:3]

        t1_points = pd.DataFrame(t1_coords, columns = ['x','y','z'])
        t1_points['label'] = df_pred_sub['labels'].reset_index(drop = True)
        df_to_fcsv(t1_points, final_fname_t1)

print('Loop 2: Labelling')  
for subject in subjects:
    print(f'{subject}') 
    
    final_fname_t1 = f'{output_dir}/{subject}_res-0p4mm_desc-space-T1w_unet.fcsv'
    
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

        df_to_fcsv(labeled_contacts,f'{output_dir}/{subject}_res-0p4mm_desc-space-T1w_unet.fcsv')

