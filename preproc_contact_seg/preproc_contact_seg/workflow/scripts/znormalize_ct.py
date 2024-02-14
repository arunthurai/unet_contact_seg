#!/usr/bin/env python
# coding: utf-8

import numpy as np
import nibabel as nib
def read_nii_metadata(nii_path):
    """Load in nifti data and header information"""
    nii = nib.load(nii_path)
    nii_affine = nii.affine
    nii_data = nii.get_fdata()
    nii_header = nii.header
    return nii_affine, nii_data, nii_header

def z_normalization(nii_path, znorm_path):
    aff, nii_img, head = read_nii_metadata(nii_path)
    img_stdev = np.std(nii_img)
    img_mean = np.mean(nii_img)
    z_norm = (nii_img - img_mean)/img_stdev
    new_img = nib.Nifti1Image(z_norm, aff, head)
    nib.save(new_img, znorm_path)

if __name__ == "__main__":
    z_normalization(
        nii_path= snakemake.input['resampled_ct'],
        znorm_path= snakemake.output['znorm_ct']
    )