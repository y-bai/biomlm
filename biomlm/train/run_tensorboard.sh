# 
# (C)Copyright 2023-2024: Yong Bai, baiyong at genomics.cn
#

log_dir=/home/share/huadjyin/home/baiyong01/projects/biomlm/biomlm/outputs/T2T_BPE_50010_1024/log

# protobuf version need 5.26.0rc2
# https://github.com/googleapis/proto-plus-python/issues/431
tensorboard --logdir ${log_dir} --port 4122