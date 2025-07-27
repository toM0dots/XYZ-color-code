#!/bin/bash
#SBATCH --array=1
#SBATCH --job-name="UH-N18"
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=128
#SBATCH --mem=35g
#SBATCH --tmp=20g
#SBATCH --time=20:00:00
#SBATCH -p msilarge
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tang1014@umn.edu
#SBATCH --output="slurm_out/%A-%a/output.txt"

#load and activate local python environment
module load conda
source activate decoder_env
cd ~/XYZ-color-code/XYZ_color_code

WORKDIR=slurm_out/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}

######------------------------------------------------------
seeds=$(seq 0 2000)
# seeds=1
error_rate=0.0001
H=12
L=9
#######################################################################

python3 benchmark.py --H ${H} --L ${L} --error_rate ${error_rate} --seeds ${seeds}
# python3 test.py




