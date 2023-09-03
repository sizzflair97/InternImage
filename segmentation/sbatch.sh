#!/bin/bash
#SBATCH --nodes=1           # set #nodes to use
#SBATCH --partition=gpu2    # set partition to use
#SBATCH --gres=gpu:a10:4   # set types and number of gpu
#SBATCH --cpus-per-task=4  # set number of cpu cores
#SBATCH -o ./_out/%j.%N.out # log file. %j: JOBID, %N: nodename
#SBATCH -e ./_err/%j.%N.err # err log file.
#================================================
 
echo "start at:" `date`     # job start time
echo "node: $HOSTNAME"      # node job started from
echo "jobid: $SLURM_JOB_ID" # print job id

sh dist_train.sh configs/ade20k/upernet_internimage_h_896_160k_ade20k.py 4

lsb_release -a