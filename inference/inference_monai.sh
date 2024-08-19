#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00


echo "begin"
source $(kpy _kpy_wrapper)

nvidia-smi

kpy load monai_unet

module load cuda cudnn

echo "beginning inference"

python3 /project/6050199/athurai3/unet_contact_seg/inference/monai_inference_pnms.py
