#coding: utf-8
"""
    Convulotional Pose Machine
        For Single Person Pose Estimation
    Human Pose Estimation Project in Lab of IP
    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
    Experimental Code
        !!DO NOT USE IT AS DEPLOYMENT!!
"""
import os
import time
import Global
import numpy as np
import tensorflow as tf
import PoseNet

class CPM(PoseNet.PoseNet):
    """
    CPM net
    """
    def __init__(self, *args, **kwargs):
        """ CPM Net implemented with Tensorflow

        directly inherit the __init__ function of superclass 'PoseNet'
        """
        super(CPM, self).__init__(*args, **kwargs)


    def net(self, image, name='CPM'):
        """ CPM Net Structure
        Args:
            image           : Input image with n times of 8
                                size:   batch_size * in_size * in_size * sizeof(RGB)
        Return:
            stacked heatmap : Heatmap NSHWC format
                                size:   batch_size * stage_num * in_size/8 * in_size/8 * joint_num
        """
        with tf.variable_scope(name):
            fmaps = self._feature_extractor(image, 'VGG', 'Feature_Extractor')
            stage = [None] * self.stage
            stage[0] = self._cpm_stage(fmaps[-1], 1, None)
            #stage[0] = tf.expand_dims(self._cpm_stage(fmap, 1, None), axis=1)

            # CPM Modules
            for t in range(2, self.stage + 1):
                stage[t-1] = self._cpm_stage(fmaps[-1], t, stage[t-2])
            
            # Ensure the consistency between pooling scale and number of feature maps
            if self.num_decoder > 0:
                assert len(fmaps) >= self.num_decoder, 'Limit with the number of returned feature maps!'
                decoded = [None] * self.num_decoder
                # Decoder Modules
                fmaps.reverse()
                for t in range(self.num_decoder):
                    if t==0:
                        decoded[t] = self._decode(fmaps[t], t+1, stage[-1])
                    else:
                        decoded[t] = self._decode(fmaps[t], t+1, decoded[t-1])   
                
                return [tf.nn.sigmoid(one_stage, name='final_output_stage%d' %(i+1))
                    for i, one_stage in enumerate(stage)], [tf.nn.sigmoid(decoded_map, name='final_decoded_map%d' %(j+1))
                    for j, decoded_map in enumerate(decoded)]
            else:
                return [tf.nn.sigmoid(one_stage, name='final_output_stage%d' %(i+1))
                    for i, one_stage in enumerate(stage)]
            

    def _feature_extractor(self, inputs, net_type='VGG', name='Feature_Extractor'):
        """ Feature Extractor
        For VGG Feature Extractor down-scale by x8
        For ResNet Feature Extractor downscale by x8 (Current Setup)

        Net use VGG as default setup
        Args:
            inputs      : Input Tensor (Data Format: NHWC)
            name        : Name of the Extractor
        Returns:
            net         : Output Tensor            
        """
        with tf.variable_scope(name):
            if net_type == 'ResNet':
                net = self._conv_bn_relu(inputs, 64, 7, 2, 'SAME')
                #   down scale by 2
                net = self._residual(net, 128, 'r1')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool1')
                #   down scale by 2
                net = self._residual(net, 128, 'r2')
                net = self._residual(net, 256, 'r3')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool2')
                #   down scale by 2
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool3')
                net = self._residual(net, 512, 'r4')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool4')
                #   optional 
                #net = self._residual(net, 512, 'r5')
                return net
            else:
                #   VGG based
                net = self._conv_bn_relu(inputs, 8, 3, 1, 'SAME', 'conv1_1', lock=not self.training)
                #   down scale by 2
                net = self._conv_bn_relu(net, 8, 3, 1, 'SAME', 'conv1_2', lock=not self.training) #16
                net_2 = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self._conv_bn_relu(net_2, 16, 3, 1, 'SAME', 'conv2_1', lock=not self.training) #32
                net_4 = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool2')
                """
                #   down scale by 2
                net = self._conv_bn_relu(net_4, 32, 3, 1, 'SAME', 'conv3_1', lock=not self.training)
                net_8 = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool3')
                """
                return [net_2, net_4]


    def _cpm_stage(self, feat_map, stage_num, last_stage=None):
        """ CPM stage Sturcture
        Args:
            feat_map    : Input Tensor from feature extractor
            last_stage  : Input Tensor from below
            stage_num   : stage number
        """

        with tf.variable_scope('CPM_stage'+str(stage_num)):
            if stage_num == 1:
                net = self._conv_bn_relu(feat_map, 32, 3, 1, 'SAME', 'conv4_3_CPM', lock=not self.training)  #32->16
                net = self._conv_bn_relu(net, 32, 3, 1, 'SAME', 'conv4_4_CPM', lock=not self.training)
                net = self._conv_bn_relu(net, 32, 3, 1, 'SAME', 'conv4_5_CPM', lock=not self.training)

                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Seg_stage'+str(stage_num), lock=not self.training)

            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], axis=3)

                # net = self._atrousconv_bn_relu(net, 16, 7, 1, 'SAME', 'Mconv1_stage'+str(stage_num), lock=not self.training)  #16->8
                # net = self._atrousconv_bn_relu(net, 16, 7, 1, 'SAME', 'Mconv2_stage'+str(stage_num), lock=not self.training)
                net = self._conv_bn_relu(net, 16, 5, 1, 'SAME', 'Mconv1_stage'+str(stage_num), lock=not self.training)  #16->8
                net = self._conv_bn_relu(net, 16, 5, 1, 'SAME', 'Mconv2_stage'+str(stage_num), lock=not self.training)
               
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Seg_stage'+str(stage_num), lock=not self.training)
        return net
    
    def _decode(self, feat_map, sample_num, last_stage):
        """ Decoder module
        Args:
            feat_map    : Input Tensor from feature extractor (unnecessary to be the last layer)
            last_stage  : Input Tensor from below (usually the last predicted heat map)
            sample_num  : stage number in the whole upsampling decoder
        """
        
        with tf.variable_scope('Decoder_stage'+str(sample_num)):
            net = tf.concat([feat_map, last_stage], axis=3)
            
            net = self._deconv_bn_relu(net, 16, 3, 2, 'SAME', 'Mdeconv1_decoder'+str(sample_num), lock=not self.training)
            
            net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Seg_decoder'+str(sample_num), lock=not self.training)
        return net
