#!/bin/bash
#DSUB -n job_biomlm
#DSUB -A root.project.P20Z10200N0170
#DSUB -R 'cpu=20;gpu=4;mem=50000'
#DSUB -N 2
#DSUB -eo tmp/%J.%I.err
#DSUB -oo tmp/%J.%I.out

## Set scripts
RANK_SCRIPT="run_multi_node_gpu_script.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/train"

## Set NNODES
NNODES=2

## Create nodefile
JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
#cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/tmp/${JOB_ID}.nodefile
# cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > tmp/CCS_ALLOC_FILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}