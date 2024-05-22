#!/bin/bash
#SBATCH --account=ctb-akhanf
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2-00:00

source /project/6050199/athurai3/unet_contact_seg/preproc_contact_seg/preproc_contact_seg/.venv/bin/activate

module load apptainer

python /project/6050199/athurai3/unet_contact_seg/preproc_contact_seg/preproc_contact_seg/run.py '/project/6050199/athurai3/seeg_data_final' '/scratch/athurai3/preproc_final' participant --profile cc-slurm --verbose --singularity-args="-B /scratch/athurai3" -c4