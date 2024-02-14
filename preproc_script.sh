#!/bin/bash

source /project/6050199/athurai3/unet_contact_seg/preproc_contact_seg/preproc_contact_seg/.venv/bin/activate

python /project/6050199/athurai3/unet_contact_seg/preproc_contact_seg/preproc_contact_seg/run.py /scratch/athurai3/seeg_data/atlasreg/ /scratch/athurai3/preproc_outputs participant --profile cc-slurm --singularity-args="-B /scratch/athurai3" -c4
