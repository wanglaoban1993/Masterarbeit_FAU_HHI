#!/bin/bash

#SBATCH --job-name=test

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<tianqi.wang@hhi.fraunhofer.de>

#SBATCH --output=%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

#SBATCH --gpus=2

#SBATCH --mem=32G

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

echo "Job with base_env_pytorch.sif env, 
+ here is running train.py in ddpm for sudoku [9,9,9]"

#cp -r ${SLURM_SUBMIT_DIR}/sudoku ${LOCAL_JOB_DIR}

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used:
# - One for the ImageNet dataset and
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node

# #apptainer run --nv --bind $DATAPOOL1/datasets:/mnt/datasets ./ddsm_image.sif
# apptainer exec --nv --bind ${LOCAL_JOB_DIR} \
# ../base_env_pytorch.sif \
# #/projects/MA_2024.sif \
# python3 ${SLURM_SUBMIT_DIR}/presample_noise.py -n 50000 -c 9 -t 400 --max_time 1 --out_path sudoku/\

# Run Apptainer with NVIDIA GPU support and activate conda environment
apptainer exec --nv --bind ${LOCAL_JOB_DIR} ./env_sudoku_ddpm.sif \
bash -c "
source /opt/conda/bin/activate base  # Activate the base conda environment
python3 ${SLURM_SUBMIT_DIR}/predict.py
"
#python3 ${SLURM_SUBMIT_DIR}/sudoku/train_sudoku_reflect_NO_sde100.py

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding th$
cp -r ${LOCAL_JOB_DIR}/${SLURM_JOB_ID} ${SLURM_SUBMIT_DIR}
mv ${SLURM_JOB_ID}* out/
rm -r ${LOCAL_JOB_DIR}/sudoku

echo "Job finished" 
