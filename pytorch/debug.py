#
#
#
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#
# from data_generator import DataGenerator, TestDataGenerator
# from utilities import (create_folder, get_filename, create_logging,
#                        calculate_confusion_matrix, calculate_accuracy,
#                        plot_confusion_matrix, print_accuracy,
#                        write_leaderboard_submission, write_evaluation_submission)
# from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling, FGSMAttack, ResNet, Vggish
# import config
from torch.autograd import Variable
#
# Model = Vggish # ResNet #DecisionLevelMaxPooling
#
#
# dataset_dir="/Users/liuhongye/BIT/Adversarial_Attacks_for_SER/data"
#
# workspace="/Users/liuhongye/BIT/Adversarial_Attacks_for_SER/data/code/pub_demos_cnn"
#
# subdir="development-subtaskA"
#
# holdout_fold=1
#
# # os.makedirs(os.path.join(dataset_dir, subdir, 'evaluation_setup'))
# dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
#                              'fold{}_devel.txt'.format(holdout_fold))
#
# dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
#                                 'fold{}_test.txt'.format(holdout_fold))
# batch_size = 16
# feature_type = "logmel"
# hdf5_path = os.path.join(workspace, 'features', feature_type, 'development.h5')
#
# print(hdf5_path)
# print(dev_train_csv)
# print(dev_validate_csv)
#
# generator = DataGenerator(hdf5_path=hdf5_path,
#                           batch_size=batch_size,
#                           dev_train_csv=dev_train_csv,
#                           dev_validate_csv=dev_validate_csv)
# #
# # for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):
# #     print()
#
# model = Model(2)
# adversary = FGSMAttack(epsilon=0.1, alpha=0.05)
#
#
# def forward(model, model_adv, generate_func, cuda, return_target):
#     """Forward data to a model.
#
#     Args:
#       generate_func: generate function
#       cuda: bool
#       return_target: bool
#
#     Returns:
#       dict, keys: 'audio_name', 'output'; optional keys: 'target'
#     """
#
#     outputs = []
#     audio_names = []
#     outputs_adv = []
#
#     if return_target:
#         targets = []
#
#     # Evaluate on mini-batch
#     for data in generate_func:
#
#         if return_target:
#             (batch_x, batch_y, batch_audio_names) = data
#
#         else:
#             (batch_x, batch_audio_names) = data
#
#         batch_x = move_data_to_gpu(batch_x, cuda)
#
#         # Predict
#         model.eval()
#         batch_output = model(batch_x)
#
#         # advesarial predict
#         batch_y_pred = np.argmax(batch_output.data.cpu().numpy(), axis=-1)
#         batch_y_pred = move_data_to_gpu(batch_y_pred, cuda)
#
#         model_cp = copy.deepcopy(model)
#         for p in model_cp.parameters():
#             p.requires_grad = False
#         model_cp.eval()
#
#         model_adv.model = model_cp
#         del model_cp
#
#         batch_x_adv = model_adv.perturb(batch_x.data.cpu().numpy(), batch_y_pred.data.cpu().numpy(), cuda=cuda)
#         batch_x_adv = move_data_to_gpu(batch_x_adv, cuda)
#         batch_output_adv = model(batch_x_adv)
#
#         # Append data
#         outputs.append(batch_output.data.cpu().numpy())
#         audio_names.append(batch_audio_names)
#         outputs_adv.append(batch_output_adv.data.cpu().numpy())
#
#         if return_target:
#             targets.append(batch_y)
#
#     dict = {}
#
#     outputs = np.concatenate(outputs, axis=0)
#     dict['output'] = outputs
#
#     audio_names = np.concatenate(audio_names, axis=0)
#     dict['audio_name'] = audio_names
#
#     outputs_adv = np.concatenate(outputs_adv, axis=0)
#     dict['output_adv'] = outputs_adv
#
#     if return_target:
#         targets = np.concatenate(targets, axis=0)
#         dict['target'] = targets
#
#     return dict
#
#     # Generate function
# generate_func = generator.generate_validate(data_type='validate',
#                                             devices=['a'],
#                                             shuffle=True,
#                                             max_iteration=None)
#
# # Forward
# dict = forward(model=model,
#                model_adv=adversary,
#                generate_func=generate_func,
#                cuda=False,
#                return_target=True)
#
# outputs = dict['output']  # (audios_num, classes_num)
# print(outputs)
# outputs_adv = dict['output_adv']  # (audios_num, classes_num)
# targets = dict['target']  # (audios_num, classes_num)
#
# predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
# predictions_adv = np.argmax(outputs_adv, axis=-1)  # (audios_num,)
#
# a = 1+1

# torch_predic = Variable(torch.Tensor([1,1]))
# torch_label = Variable(torch.LongTensor([1,1]))
# a = F.nll_loss(torch_predic, torch_label)
# print(a)

def calculate_recall_specificity_Macc(confusion_matrix):
    """
    Calculates recall and specificity from a confusion matrix.

    Parameters:
    confusion_matrix (numpy.matrix): A square matrix of size m x m where m is the number of classes.

    Returns:
    recall (numpy.float64): The recall value.
    specificity (numpy.float64): The specificity value.
    """
    # Calculate the sum of true positives and false negatives for each class
    tp_fn = np.sum(confusion_matrix, axis=1)

    # Calculate the sum of true positives and false positives for each class
    tp_fp = np.sum(confusion_matrix, axis=0)

    # Calculate the true positives for each class
    tp = np.diag(confusion_matrix)

    # Calculate the false negatives for each class
    fn = tp_fn - tp

    # Calculate the false positives for each class
    fp = tp_fp - tp

    # Calculate the true negatives for each class
    tn = np.sum(confusion_matrix) - (tp_fn + tp_fp - tp)

    # Calculate recall and specificity for each class
    recall = tp / tp_fn
    specificity = tn / (tn + fp)
    Macc = (recall+specificity)/2

    return recall, specificity, Macc


con = np.array([[483,  20],[ 41,  85]])
print(calculate_recall_specificity_Macc(con))