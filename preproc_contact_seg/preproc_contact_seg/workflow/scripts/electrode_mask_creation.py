#!/usr/bin/env python
# coding: utf-8


import os
import nibabel as nib
import sys
import pandas as pd 
import numpy as np
import subprocess
from glob import glob
from skimage.morphology import dilation
from skimage.morphology import ball
from scipy.ndimage import gaussian_filter
import regex as re
import csv


def determineFCSVCoordSystem(input_fcsv,overwrite_fcsv=False):
	# need to determine if file is in RAS or LPS
	# loop through header to find coordinate system
	coordFlag = re.compile('# CoordinateSystem')
	verFlag = re.compile('# Markups fiducial file version')
	headFlag = re.compile('# columns')
	coord_sys=None
	headFin=None
	ver_fin=None
	
	with open(input_fcsv, 'r') as myfile:
		firstNlines=myfile.readlines()[0:3]
	
	for row in firstNlines:
		row=re.sub("[\s\,]+[\,]","",row).replace("\n","")
		cleaned_dict={row.split('=')[0].strip():row.split('=')[1].strip()}
		if None in list(cleaned_dict):
			cleaned_dict['# columns'] = cleaned_dict.pop(None)
		if any(coordFlag.match(x) for x in list(cleaned_dict)):
			coord_sys = list(cleaned_dict.values())[0]
		if any(verFlag.match(x) for x in list(cleaned_dict)):
			verString = list(filter(verFlag.match,  list(cleaned_dict)))
			assert len(verString)==1
			ver_fin = verString[0].split('=')[-1].strip()
		if any(headFlag.match(x) for x in list(cleaned_dict)):
			headFin=list(cleaned_dict.values())[0].split(',')
	return coord_sys


def extract_coords(file_path):
    coord_sys=determineFCSVCoordSystem(file_path)
    df = pd.read_csv(file_path, skiprows = 3, header = None)
    coord_arr = df[[1,2,3]].to_numpy()
    if any(x in coord_sys for x in {'LPS','1'}):
        coord_arr = coord_arr * np.array([-1,-1,1])
    return coord_arr

def transform_points_ct_space(coord_path, ct_t1_path):
    t1_coords = extract_coords(coord_path)
    ct_coords = np.zeros(t1_coords.shape)
    transform = np.loadtxt(ct_t1_path)
    transform = np.linalg.inv(transform)

    M = transform[:3,:3]
    abc = transform[:3,3]


    for i in range(len(ct_coords)):
        vec = t1_coords[i,:]
        tvec = M.dot(vec) + abc
        ct_coords[i,:] = tvec[:3]

    ct_coords = np.round(ct_coords).astype(int)

    return ct_coords

def read_nifti(img_path):
    test_img = nib.load(img_path)
    test_data = np.asarray(test_img.dataobj)
    test_aff = test_img.affine
    test_shape = test_img.shape

    return test_img, test_data, test_aff, test_shape

# https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/
def create_line_mask(point1, point2, shape):
    # Create an empty mask with the specified shape
    mask = np.zeros(shape, dtype=bool)
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    mask[x1, y1, z1] = True
    # Get the directions of each axis
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            mask[x1, y1, z1] = True
        mask[x1, y1, z1] = True

    # Driving axis is Y-axis"
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            mask[x1, y1, z1] = True
        mask[x1, y1, z1] = True

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            mask[x1, y1, z1] = True
        mask[x1, y1, z1] = True
    return mask


def transform_coords_vox_space(planned_coords, img_aff):
    inv_affine = np.linalg.inv(img_aff)
    M = inv_affine[:3,:3]
    abc = inv_affine[:3,3]

    transform_coords = np.zeros(planned_coords.shape)

    for i in range(len(transform_coords)):
        vec = planned_coords[i,:]
        tvec = M.dot(vec) + abc
        transform_coords[i,:] = tvec[:3]
    
    transform_coords = np.round(transform_coords).astype(int)

    return transform_coords

def check_coord_dims(pointa, pointb, img_size):
    for i in range(3):
        if pointa[i] > img_size[i] or pointb[i] > img_size[i]:
            print('review planned_fcsv - dim mismatch')
            return False
    return True

def create_electrode_mask(img_path, coord_path, ct_t1_trans, final_path):
    img, img_data, img_aff, img_shape = read_nifti(img_path)

    ct_coord_arr = transform_points_ct_space(coord_path, ct_t1_trans)

    transformed_coords = transform_coords_vox_space(ct_coord_arr, img_aff)
                   
    final_mask = np.zeros(img_shape).astype(bool)
    
    for i in range(0,len(transformed_coords),2):
        pointa = transformed_coords[i, :]
        pointb = transformed_coords[i+1, :]
        
        
        test_mask = create_line_mask(pointa, pointb, img_shape)
                
        footprint = ball(4)
        dilated = dilation(test_mask, footprint)
        result = gaussian_filter(dilated.astype(float), sigma=0.6)
        final_mask += result>0

    clipped_img = nib.Nifti1Image(final_mask, img_aff, img.header)
    nib.save(clipped_img, final_path)


if __name__ == "__main__":
    create_electrode_mask(
        img_path=snakemake.input['znorm_ct'],
        coord_path = snakemake.input['actual'],
        ct_t1_trans=snakemake.input['transform_matrix'],
        final_path=snakemake.output["electrode_mask"],
    )

