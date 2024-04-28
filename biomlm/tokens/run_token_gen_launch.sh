#!/bin/bash
#DSUB -n job_biomlm
#DSUB -A root.project.P24Z10200N0985
#DSUB -R 'cpu=10;gpu=0;mem=100000'
#DSUB -N 4
#DSUB -eo %J.%I.err.log
#DSUB -oo %J.%I.out.log

## Set scripts
RANK_SCRIPT="run_token_gen_script.sh"

###Set Start Path
JOB_PATH="/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/tokens"

## Set NNODES
NNODES=4

## Create nodefile
JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
#cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/tmp/${JOB_ID}.nodefile
# cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > CCS_ALLOC_FILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODEFILE}