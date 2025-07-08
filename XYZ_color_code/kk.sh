#!/bin/bash
#SBATCH --array=6
#SBATCH --job-name="UH-N18"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=125
#SBATCH --mem=20gb
#SBATCH --time=5:00:00
#SBATCH -p msismall
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tang1014@umn.edu
#SBATCH --output="slurm_out/%A-%a/output.txt"

#load and activate local python environment
module load conda
source activate decoder_env
cd ~/XYZ_color_code

WORKDIR=slurm_out/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

######################################################################
seeds= $(seq 0 2 1999)
error_rate=0.0002
H= 12
L= 9
#######################################################################

python3 benchmark.py --H ${H} --L ${L} --error_rate ${error_rate} --seeds ${seeds}



