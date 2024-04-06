#!/bin/bash
mkdir -p ./_log/$SLURM_JOB_ID
set -x

FOLDERNAME=`date +"%m-%d-%H-%M-%S"`
mkdir work_dirs/${FOLDERNAME}
# FOLDERNAME="09-21-14-09-10"

# PARTITION=$1
PARTITION="gpu2"
# JOB_NAME=$2
JOB_NAME="InternImage"
# CONFIG=$3
# CONFIG="./configs/dacon/segformer_internimage_xl_512x1024_160k_dacon.py"
# CONFIG="./configs/dacon/segformer_internimage_xl_512x1024_160k_mapillary_dacon.py"
CONFIG="./configs/dacon/upernet_internimage_xl_512x1024_80k_mapillary.py"
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-14}
# SRUN_ARGS=${SRUN_ARGS:-""}
# SRUN_ARGS="--nodes=4 --nodelist=n060,n061,n062,n082"
SRUN_ARGS="--nodes=4"
PY_ARGS=${@:4}

MONITOR_GPU_SCRIPT=$(cat <<EOF
    hostnode=\`hostname -s\`
    /usr/local/bin/gpustat -i > ./_log/\$hostnode.gpu &
EOF
)

# SRUN_SCRIPT=$(cat <<EOF
#     $MONITOR_GPU_SCRIPT

#     python -u train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} --work-dir work_dirs/${FOLDERNAME} --resume-from work_dirs/${FOLDERNAME}/latest.pth
# EOF
# )

SRUN_SCRIPT=$(cat <<EOF
    $MONITOR_GPU_SCRIPT

    python -u train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} --work-dir work_dirs/${FOLDERNAME}
EOF
)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -o ./_out/%j.out \
    -e ./_err/%j.err \
    --job-name=${JOB_NAME} \
    --gres=gpu:a10:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    bash -c "$SRUN_SCRIPT"


