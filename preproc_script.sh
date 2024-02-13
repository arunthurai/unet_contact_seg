#!/bin/bash

source /scratch/athurai3/seeg_contacts_loc/preproc_env/bin/activate

python /scratch/athurai3/seeg_contacts_loc/preproc_contact_seg/preproc_contact_seg/run.py /scratch/athurai3/seeg_contacts_loc/atlasreg/ /scratch/athurai3/preproc_outputs participant --profile cc-slurm --singularity-args="-B /scratch/athurai3"
