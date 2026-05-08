#!/bin/bash
#SBATCH -J MUD3                         # 作业名为 cut_bert_test.out
#SBATCH -o ./out/early_lstm_han_1e-5_20_30.out                           # 屏幕上的输出文件重定向到 cut_bert_test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 1:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1 # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点

#8rJ<$sEh

# source /home/bichenwang/miniconda3/etc/profile.d/conda.sh
# conda activate ccac-baseline

python main_early_ori.py --data_dir '/home/yfhou/CCAC-baseline/MUD3_final' \
               --model 'LSTM_han'  \
               --epochs 20 \
               --batch_size 2 \
               --learning_rate 1e-5 \
               --lr_scheduler None \
               --device cuda \
               --lant_save_name 'LSTM_han' \
               --earliness_alpha 30 \
               --seed 110 \