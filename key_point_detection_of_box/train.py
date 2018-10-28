#-*-coding:utf-8 -*-
from CPM import CPM
import Global
import datagen_mpii as datagen
from tensorflow.python.framework import graph_util
import tensorflow as tf
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#   Thanks to wbenhibi@github
#   good datagen to use

print('--Creating Dataset')
dataset = \
    datagen.DataGenerator(input_color=Global.input_color,
                          joints_name=Global.joint_list,
                          img_train_dir=Global.IMG_TRAIN_ROOT,
                          img_valid_dir=Global.IMG_VALID_ROOT,
                          label_train_dir=Global.LABEL_TRAIN_ROOT,
                          label_valid_dir=Global.LABEL_VALID_ROOT,
                          # train_data_file=Global.training_txt_file,
                          # valid_folder=Global.valid_folder,
                          remove_joints=None,
                          in_size_h=Global.INPUT_SIZE_H,
                          in_size_w=Global.INPUT_SIZE_W,
                          aug_data=Global.AUG_DATA,
                          ann_size_h=Global.ANN_SIZE_H,
                          ann_size_w=Global.ANN_SIZE_W,
                          Gaussian_r=Global.GAUSSIAN_R,
                          decode_scale=Global.DECODE_SCALE)

dataset._create_train_table()
dataset._randomize()
dataset._create_sets()

model = CPM(input_color=Global.input_color,
            base_lr=Global.base_lr,
            in_size_w=Global.INPUT_SIZE_W,
            in_size_h=Global.INPUT_SIZE_H,
            stage=Global.STAGE,
            num_decoder=Global.NUM_DECODER,
            joints=Global.joint_list,
            batch_size=Global.batch_size,
            epoch=Global.epoch,
            epoch_size=Global.epoch_size,
            dataset=dataset,
            log_dir=Global.LOGDIR,
            weight_decay=0.0001)

model.BuildModel()

model.train()
