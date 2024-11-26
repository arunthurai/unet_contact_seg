#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from statistics import NormalDist
import nibabel as nib
import matplotlib as plt
import matplotlib.pyplot as plt
import os
from skimage.morphology import dilation, ball
from scipy.ndimage import gaussian_filter
#import mne
from sklearn.linear_model import LinearRegression

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
		out_df['description'].append(str(ifid.iloc[4]))
		out_df['associatedNodeID'].append('')

	out_df=pd.DataFrame(out_df)
	out_df.to_csv(output_fcsv, sep=',', index=False, lineterminator="", mode='a', header=False, float_format = '%.3f')


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
    coord_arr = df[[1,2,3,11]].to_numpy()
    if any(x in coord_sys for x in {'LPS','1'}):
        labels = coord_arr[:, 3]
        coord_arr = coord_arr[:, :3] * np.array([-1,-1,1])
        coord_arr = np.concatenate([coord_arr, labels.reshape(-1,1)], axis=1)
    return coord_arr

def transform_coords_vox_space(scanner_coords, img_aff):
    M = img_aff[:3,:3]
    abc = img_aff[:3,3]

    transform_coords = np.zeros(scanner_coords.shape)

    for i in range(len(transform_coords)):
        vec = scanner_coords[i,:]
        tvec = M.dot(vec) + abc
        transform_coords[i,:] = tvec[:3]
    
    #transform_coords = np.round(transform_coords).astype(int)

    return transform_coords

def transform_points_t1_space(ct_coords, transform):
    t1_coords = np.zeros(ct_coords.shape)
    #transform = np.loadtxt(ct_t1_path)

    M = transform[:3,:3]
    abc = transform[:3,3]


    for i in range(len(ct_coords)):
        vec = ct_coords[i,:]
        tvec = M.dot(vec) + abc
        t1_coords[i,:] = tvec[:3]

    # ct_coords = np.round(ct_coords).astype(int) #DELETE

    return t1_coords

# Function to calculate the angle between two vectors in degrees
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def trace_line(entry_point, target_point):
    # Vector from target to entry
    return (entry_point - target_point)/np.linalg.norm(target_point - entry_point)

def find_next_contact_new(current_point,
                      direction,
                      contacts,
                      max_angle_target,
                      max_distance_prior, # percentage
                      min_distance_prior, # percentage
                      avg_distance,
                      entry_point,
                      distance_from_entry=1.5,
                        ):
    next_contact = None
    next_contact_id = None
    min_distance = float('inf')
    # print(current_point, avg_distance, direction)
    theory_point = current_point + avg_distance*direction
    # Projection of entry on direction vector
    entry_proj = np.dot(entry_point, direction) / np.linalg.norm(direction)
    
    for ind, contact_array in enumerate(contacts):
        contact = contact_array[0:3]
        
        # Vector last detected contact to possible next contact
        v_contact = trace_line(contact, current_point)
        # Angle between the predicted direction and the v_contact
        angle_contact = calculate_angle(v_contact, direction)
        distance_target = np.linalg.norm(theory_point - contact)
        distance_prior = np.linalg.norm(current_point - contact)
        distance_entry = np.linalg.norm(entry_point - contact)
        contact_proj = np.dot(contact, direction) / np.linalg.norm(direction)
        # print(ind, angle_contact, distance_prior, distance_entry)
        # print(0, max_distance_prior*avg_distance, min_distance_prior*avg_distance, distance_from_entry, '\n')
        distance_metric = np.sqrt(distance_target**2+distance_prior**2) # balance between distance to target and prior
        # Check that it's the closest point to the target, 
        # that it's below a specific distance from the target
        # and above a certain distance from the current point,
        # that it is not too close to the entry
        # and that is not beyond the entry point
        if (distance_metric < min_distance and
            angle_contact <= max_angle_target and
            distance_prior >= min_distance_prior*avg_distance and
            distance_prior <= max_distance_prior*avg_distance):
            if distance_from_entry is None or (distance_entry >= distance_from_entry and contact_proj < entry_proj):
                min_distance = distance_metric
                next_contact = contact
                next_contact_id = ind
    if next_contact_id is not None:
        # print(contacts[next_contact_id,-1], '\n')
        # print(next_contact_id, '\n')
        contacts = np.delete(contacts, next_contact_id, axis=0) 
    return next_contact, contacts

def adjust_direction_using_all_contacts(contacts, original_direction):
    n_elements_reg = 3 # use last 3 elements
    contacts = np.array(contacts)[-n_elements_reg:,:] # only use last 3 coords
    # print(contacts)
    
    # Check dimension where all vals are not equal
    dim = None
    i = 0
    dims = list(range(n_elements_reg))
    while dim is None and i<n_elements_reg:
        if not np.all(contacts[:,i] == contacts[0,i]):
            dim = dims.pop(i)
        i += 1
    assert dim is not None
    
    X = contacts[:, dim].reshape(-1, 1)
    y1 = contacts[:, dims[0]]
    y2 = contacts[:, dims[1]]

    reg1 = LinearRegression().fit(X, y1)
    reg2 = LinearRegression().fit(X, y2)

    direction_vector = np.ones(n_elements_reg)
    direction_vector[dims[0]] = reg1.coef_[0]
    direction_vector[dims[1]] = reg2.coef_[0]
    # Normalize
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Ensure the direction vector points toward the initial direction
    if np.dot(direction_vector, original_direction) < 0:
        direction_vector = -direction_vector
        
    return direction_vector

def run_segmentation_qc_new(entry_point,
                             target_point,
                             contacts, 
                             max_angle_target, 
                             max_distance_prior, 
                             min_distance_prior, 
                             initial_spacing=4,
                             max_n_contacts = -1 # set to -1 to apply distance threshold to entry
                           ):
    original_direction = trace_line(entry_point, target_point)
    direction = original_direction
    current_point = target_point
    avg_distance = 2 # mm
    
    # Initialize with the closest contact to the entry
    current_point, contacts = find_next_contact_new(current_point,
                                                    direction, 
                                                    contacts, 
                                                    90, 
                                                    2, 
                                                    0, 
                                                    avg_distance, 
                                                    entry_point)
    found_contacts = [current_point]
    distances_between_contacts = []
    next_contact = current_point
    # Find next contact
    next_contact, contacts = find_next_contact_new(current_point, 
                                                   direction, 
                                                   contacts, 
                                                   max_angle_target, 
                                                   max_distance_prior, 
                                                   min_distance_prior, 
                                                   initial_spacing, 
                                                   entry_point)
    # Calculate
    while next_contact is not None:   
        found_contacts.append(next_contact)
        distances_between_contacts.append(np.linalg.norm(next_contact - current_point))
        if len(found_contacts) >= 3:
            direction = adjust_direction_using_all_contacts(found_contacts, original_direction)
        current_point = next_contact

        avg_distance = np.mean(distances_between_contacts)
        
        # Find next contact
        if len(contacts) > 0:
            next_contact, contacts = find_next_contact_new(current_point,
                                                           direction, 
                                                           contacts, 
                                                           max_angle_target, 
                                                           max_distance_prior, 
                                                           min_distance_prior, 
                                                           avg_distance, 
                                                           entry_point,
                                                          distance_from_entry=1.5 if len(found_contacts)>=max_n_contacts else None)
        else:
            next_contact = None
    print(f"Adjusted Mean Distance: {avg_distance}")
    return found_contacts, avg_distance
    
def create_electrode_mask(ct_data: np.ndarray, entry_target_vox: np.ndarray) -> np.ndarray:
    # Create mask electrode mask using Bresenham's line algorithm
    final_mask = np.zeros(ct_data.shape).astype(bool)
    test_mask = create_line_mask(entry_target_vox[i, :], entry_target_vox[i+1, :], ct_data.shape)
    # Dilate the mask to avoid loosing potential intersection with CT
    dilated = dilation(test_mask, ball(4))
    result = gaussian_filter(dilated.astype(float), sigma=0.6)
    final_mask += result>0
    
    # Compute mask based on CT intensity
    mask_intensity = (ct_data>2500).astype(bool)

    # Merge the masks
    merged_mask = final_mask*mask_intensity

    # Dilate a bit to avoid loosing any close-by contacts
    footprint = ball(3)
    merged_mask = dilation(merged_mask, footprint)

    return merged_mask

model_desc = 'Jun17_patch95_4layers_diceCE'
pnms_dir = f'/scratch/athurai3/val_predictions/{model_desc}/prob_nms'
patch_size = 'patch96'
gt_dir = '/project/6050199/athurai3/seeg_data_final'
# test_dir = '/project/6050199/athurai3/special_electrodes'
# subjects = [identifier for identifier in os.listdir(test_dir) if "sub-" in identifier]

adtech_dict = dict(
    [(3,"RD10R-SP03X"), (4,"RD10R-SP04X"), (5,"RD10R-SP05X"), (6,"RD10R-SP06X"), (7,"RD10R-SP07X")]
)
    

subjects = []

file_path = '/project/6050199/athurai3/unet_contact_seg/data_split_thesis/val_thesis.txt'
with open(file_path, 'r') as file:
    subjects = [line.strip() for line in file if line.strip()]

print(subjects)
print(sorted(subjects))
     
for sub in sorted(subjects):
    print(sub)
    final_fname_t1 = f'{pnms_dir}/{sub}_space-T1w_desc-{patch_size}_unet_pnms.fcsv'
    orig_pnms = pd.read_csv(f'{pnms_dir}/{sub}_desc-{patch_size}_unet_pnms.fcsv', skiprows=3, header = None)
    ct_t1_trans = np.loadtxt(f'{gt_dir}/{sub}/{sub}_desc-rigid_from-ct_to-T1w_type-ras_ses-post_xfm.txt')
    
    # TODO: Update paths
    ct = nib.load(f'/scratch/athurai3/preproc_final/{sub}/{sub}_res-0p4mm_ct.nii.gz')
    entry_target_path = f'{gt_dir}/{sub}/{sub}_actual.fcsv'
    
    # Get affine to transform to voxels
    inv_affine = np.linalg.inv(ct.affine)
    data = np.asarray(ct.dataobj)

    # Get entry-target coords
    entry_target = extract_coords(entry_target_path)
    # Transform to CT space
    entry_target_coords = transform_points_t1_space(entry_target[:,:-1].astype(float), np.linalg.inv(ct_t1_trans))
    #entry_target_coords = mne.transforms.apply_trans(np.linalg.inv(ct_t1_trans), entry_target[:,:-1].astype(float))
    
    # Transform to voxels
    entry_target_vox = np.round(transform_coords_vox_space(entry_target_coords, inv_affine)).astype(int)
    #entry_target_vox = np.round(mne.transforms.apply_trans(inv_affine, entry_target_coords)).astype(int)
    
    # Coordinates outputted by the network
    coords_interest = orig_pnms.iloc[:,1:4].to_numpy()
    # Get contacts positions in voxel space
    transformed_coords = np.round(transform_coords_vox_space(coords_interest, inv_affine)).astype(int)
    
    # Empty final df
    df_pos = pd.DataFrame(columns=['x', 'y', 'z', 'label_order', 'desc'])
    df_distance = pd.DataFrame(columns=['electrode', 'avg_distance'])
    elec_labels = []
    avg_distances = []
    
    """
    First stage: Masking filtering
    """
    # electrode_mask = nib.load(f'/scratch/athurai3/preproc_final/{sub}/{sub}_res-0p4mm_desc-electrode_mask.nii.gz')
    # electrode_mask = electrode_mask.get_fdata()

    #filtered_coords = coords_interest[electrode_mask[transformed_coords[:,0], transformed_coords[:,1], transformed_coords[:,2]] > 0]
    
    filtered_coords = coords_interest
    """
    Second stage: Line filtering
    """
    # Run a first time to figure out the amount of contacts
    # Compute for each pair of entry-target
    n_contacts = []
    contacts_coords = []
    for i in range(0,entry_target.shape[0],2):  
        # Label for the contact
        label = entry_target[i,-1]
        print(label, flush=True)        
        found_contacts, avg_distance = run_segmentation_qc_new(entry_point = entry_target_coords[i+1, :],
                                                               target_point = entry_target_coords[i, :],
                                                               contacts = filtered_coords, 
                                                               max_angle_target = 35, 
                                                               max_distance_prior = 2, 
                                                               min_distance_prior = 0.5, 
                                                               initial_spacing=4)
        n_contacts.append(len(found_contacts))
        contacts_coords.append(found_contacts)
    # Find the max_n_contacts
    print(n_contacts)
    elements, count = np.unique(n_contacts, return_counts=True)
    # Flip them to order from higher to lower
    elements = np.flip(elements)
    count = np.flip(count)
    # Find cumulative sum. Indicates the number of channels with values higher than a number
    cumcount = np.cumsum(count)
    max_n_contacts = np.max(elements[np.cumsum(count)>=int(len(n_contacts)*0.3)])
    print('Number of contacts for electrode: ', max_n_contacts)
    # max_n_contacts = np.max(elements[count>1])
    # print('Number of contacts for electrode: ', max_n_contacts)
    
    # Now repeat to find the actual points
    # Compute for each pair of entry-target
    for i in range(0,entry_target.shape[0],2):  
        # Label for the contact
        label = entry_target[i,-1]
        print(label, flush=True)

        # Only re-execute if len(found_contacts)<max_n_contacts (not all contacts were found)
        found_contacts = contacts_coords[i//2]
        if len(found_contacts) < max_n_contacts:
            found_contacts, avg_distance = run_segmentation_qc_new(entry_point = entry_target_coords[i+1, :],
                                                               target_point = entry_target_coords[i, :],
                                                               contacts = filtered_coords, 
                                                               max_angle_target = 35, 
                                                               max_distance_prior = 2, 
                                                               min_distance_prior = 0.5, 
                                                               initial_spacing=4,
                                                               max_n_contacts=max_n_contacts)
        """
        Third stage: Labelling
        """
        found_contacts = np.array(found_contacts)
        elec_labels.append(label)
        avg_distances.append(avg_distance)
        t1_coords = transform_points_t1_space(found_contacts, ct_t1_trans) #t1_coords = mne.transforms.apply_trans(ct_t1_trans, found_contacts)
        df_tmp = pd.DataFrame(t1_coords, columns=['x','y','z'])
        df_tmp['label_order'] = [f'{label}-{idx+1:02d}' for idx in range(found_contacts.shape[0])]
        print('Number of contacts found: ',  found_contacts.shape[0], '\n')
        # if found_contacts.shape[0] < 8:
        #     print(f'{sub} Review', '\n')
        
        # elif found_contacts.shape[0] == 8 and np.round(avg_distance) == 5:
        #     df_tmp['desc'] = 'MM16D-SPO5X'
        # elif np.round(avg_distance, 2) == 3.5:
        #     df_tmp['desc'] = 'DIXI'
        # elif np.round(avg_distance) in adtech_dict.keys():
        #     #print(adtech_dict[np.round(avg_distance)])
        #     df_tmp['desc'] = adtech_dict[np.round(avg_distance)]
        # else:
        #     df_tmp['desc'] = 'NA'

        df_pos = pd.concat([df_pos, df_tmp])
        
    
    df_to_fcsv(df_pos, final_fname_t1)

    df_distance['electrode'] = elec_labels
    df_distance['avg_distance'] = avg_distances
    df_distance.to_csv(f'{pnms_dir}/{sub}_electrode_distance.fcsv')

    