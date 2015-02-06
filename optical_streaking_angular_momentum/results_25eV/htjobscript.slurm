#!/bin/bash -l
#Job name:
#SBATCH --job-name=MCTDHF.results_25eV

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16

#SBATCH --partition=lr3
#SBATCH --account=lr_axl
#SBATCH --qos=condo_axl


#Wall clock limit:
#SBATCH --time=5:09:00



cd $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
echo $SLURM_TASKS_PER_NODE
echo $LD_LIBRARY_PATH
echo $PATH
ht_helper.sh -t taskfile.txt -n1 -s1 -dvk
