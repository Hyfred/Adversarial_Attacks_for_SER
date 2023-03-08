#!/bin/bash
# You need to modify this path
#DATASET_DIR="/home/renzhao/demos/demos4"
DATASET_DIR="/content/drive/MyDrive/Adversarial_Attacks_for_SER/data"

# You need to modify this path as your workspace
#WORKSPACE="/home/renzhao/demos/demos4/pub_demos_cnn"
WORKSPACE="/content/drive/MyDrive/Adversarial_Attacks_for_SER/data/code/pub_demos_cnn"

#DEV_SUBTASK_A_DIR="demos_data"
DEV_SUBTASK_A_DIR="development-subtaskA"

BACKEND="pytorch-baseline"
HOLDOUT_FOLD=1
GPU_ID=0
FEATURE="logmel"
CONVLAYER=1 

############ Extract features ############
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE

############ Train and validate the baseline ############(if train in train data valida in valida data, add '--validation')
python $BACKEND/main_pytorch_train.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --alpha_value=0.1 --cuda

# Evaluate subtask A
python $BACKEND/main_pytorch_train.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --feature_type=$FEATURE --holdout_fold=$HOLDOUT_FOLD --iteration=10000 --alpha_value=0.1 --cuda


