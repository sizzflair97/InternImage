#!/bin/bash
#SBATCH --nodes=4           # set #nodes to use
#SBATCH --partition=gpu2    # set partition to use
#SBATCH --gres=gpu:a10:4   # set types and number of gpu
#SBATCH --ntasks=4   # set types and number of gpu
#SBATCH --cpus-per-task=4  # set number of cpu cores
#SBATCH -o ./_out/%j.%N.out # log file. %j: JOBID, %N: nodename
#SBATCH -e ./_err/%j.%N.err # err log file.
#================================================
DISTRIBUTED_ARGS="
    --nproc_per_node ${GPUS_PER_NODE:-4} \
    --nnodes ${NNODES:-4} \
    --node_rank $2 \
    --master_addr $1 \
    --master_port ${MASTER_PORT:-6007}
"

echo "start at:" `date`     # job start time
echo "node: $HOSTNAME"      # node job started from
echo "current_addr: $(hostname -i)"
echo "jobid: $SLURM_JOB_ID" # print job id

torchrun $DISTRIBUTED_ARGS train.py configs/dacon/upernet_internimage_h_896_160k_dacon.py\
            --launcher="pytorch" \
#            --eval mIoU \

lsb_release -a