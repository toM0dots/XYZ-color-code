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

# fe_arr=( 0.995 0.99 0.98 0.95 0.85 0.75 0 )
# transient_arr=( 2000 2000 1000 650 300 300 1000 )
# step_arr=( 1 1 2 2 2 1 5 )
# num_samples_arr=( 100 1 20 20 20 20 20 )

# fe=${fe_arr[${SLURM_ARRAY_TASK_ID}]}
# transient=${transient_arr[${SLURM_ARRAY_TASK_ID}]}
# step=${step_arr[${SLURM_ARRAY_TASK_ID}]}
# num_samples=${num_samples_arr[${SLURM_ARRAY_TASK_ID}]}
######------------------------------------------------------
seeds= $(seq 0 2 1999)
error_rate=0.0002
H= 12
L= 9
#######################################################################

python3 benchmark.py --H ${H} --L ${L} --error_rate ${error_rate} --seeds ${seeds}

# python3 run_unheralded.py -Nx ${Nx} -Ny ${Ny} -Nr ${Nr} -num_proc ${SLURM_CPUS_PER_TASK} -Ns ${Ns} -num_samples ${num_samples} -fe ${fe} -eta ${eta} -gamma_x ${gamma_x} -gamma_z ${gamma_z} -rn_start ${rn_start} -slurm_dir ${WORKDIR} -transient ${transient} -step ${step}




