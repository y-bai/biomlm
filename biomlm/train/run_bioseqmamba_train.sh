# 
# (C)Copyright 2023-2024: Yong Bai, baiyong at genomics.cn

nohup torchrun --nproc_per_node 4 run_bioseqmamba_train.py > train.log &

# if use one GPU, set os.environ["CUDA_VISIBLE_DEVICES"] = "0" in front of .py script.
# nohup python run_bioseqmamba_causal.py > train.log &