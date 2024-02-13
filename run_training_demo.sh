#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --gpus-per-node=t4:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
echo "begin"
cd

source $(kpy _kpy_wrapper)

kpy load monai_unet

echo "beginning training"

python3 projects/ctb-akhanf/athurai3/unet_contact_seg/demo_monai_segmentation_3d.py
