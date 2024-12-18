#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00


echo "begin"
source $(kpy _kpy_wrapper)

echo "Copying Files to SLURM_TMPDIR"
cp '/scratch/athurai3/train_0p4mm/train_desc-35mm_patches.dat' '/scratch/athurai3/val_0p4mm/val_desc-35mm_patches.dat' $SLURM_TMPDIR

echo "Files Copied to SLURM_TMPDIR"
nvidia-smi

module load cuda cudnn

kpy load monai_unet

echo "beginning training"

python3 projects/ctb-akhanf/athurai3/unet_contact_seg/monai_c3d_implementation/monai_c3d_dice.py
