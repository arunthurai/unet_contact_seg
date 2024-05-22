#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00


echo "begin"
source $(kpy _kpy_wrapper)

echo "Copying Files to SLURM_TMPDIR"
cp '/scratch/athurai3/train_final/train_patches95.dat' '/scratch/athurai3/val_final/val_patches95.dat' $SLURM_TMPDIR

echo "Files Copied to SLURM_TMPDIR"
nvidia-smi

module load cuda cudnn

kpy load monai_unet

echo "training with patch size 96, diceCE loss"
echo "beginning training"

python3 /scratch/athurai3/sandbox_contact_seg/monai_c3d_dice.py
