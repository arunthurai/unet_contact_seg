#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00


echo "begin"
#source $(kpy _kpy_wrapper)

nvidia-smi

module load cuda cudnn

source /scratch/athurai3/sandbox_contact_seg/.monai_hdt/bin/activate

echo "beginning inference"

python3 /scratch/athurai3/sandbox_contact_seg/monai_inference_pnms.py
