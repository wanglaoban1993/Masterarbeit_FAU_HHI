#!/bin/bash

#SBATCH --job-name=test

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<tianqi.wang@hhi.fraunhofer.de>

#SBATCH --output=%j_%x.out

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

#SBATCH --gpus=1

#SBATCH --mem=32G

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

echo "Job with base_env_pytorch.sif env, 
+ here is running sudokudataset_DIY.py to extract sudoku solution from .csv to .pickle"


# Run Apptainer with NVIDIA GPU support and activate conda environment
apptainer exec --nv --bind ${LOCAL_JOB_DIR} ../base_env_pytorch.sif \
bash -c "
source /opt/conda/bin/activate base  # Activate the base conda environment
python3 ${SLURM_SUBMIT_DIR}/sudokudataset_DIY.py
"

# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding th$
cp -r ${LOCAL_JOB_DIR}/${SLURM_JOB_ID} ${SLURM_SUBMIT_DIR}
mv ${SLURM_JOB_ID}* out/
rm -r ${LOCAL_JOB_DIR}/sudoku

echo "Job finished" 
