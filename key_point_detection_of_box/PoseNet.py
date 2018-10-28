#coding: utf-8
"""
    PoseNet
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
import numpy as np
import tensorflow as tf
import Global
from tensorflow.python.framework import graph_util
# import PoseNet
import datagen_mpii

def f1():
    datagen_mpii.guass_r = Global.GAUSSIAN_r
    return 1
def f2():
    datagen_mpii.guass_r = Global.GAUSSIAN_R
    return 0

class PoseNet(object):
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size_w=384, in_size_h=200, out_size_w=None, out_size_h=None, cpu_only=False,
                 input_color='GRAY', batch_size=16, epoch=20, dataset=None, log_dir=None, stage=6, num_decoder=3,
                 epoch_size=1000, w_summary=True, training=True, weight_decay=0.0001, joints=Global.joint_list, pretrained_model=None):
        """

        :param base_lr:             starter learning rate
        :param in_size_*:           input image size
        :param out_size_*:          output image size
        :param cpu_only             use CPU or GPU
        :param input_color:         'GRAY' or 'RGB'
        :param batch_size:          size of each batch
        :param epoch:               num of epoch to train
        :param dataset:             *datagen* class to gen & feed data
        :param log_dir:             log directory
        :param stage:               num of stage in cpm model
        :param epoch_size:          size of each epoch
        :param w_summary:           bool to determine if do weight summary
        :param training:            bool to determine if the model trains
        :param weight_decay:        scale the l2 regularization loss
        :param joints:              list to define names of joints
        :param pretrained_model:    Path to pre-trained model
        :param load_pretrained:     bool to determine if the net loads all arg

        ATTENTION HERE:
        *   if load_pretrained is False
            then the model only loads VGG part of arguments
            if true, then it loads all weights & bias

        *   if log_dir is None, then the model won't output any save files
            but PLEASE DONT WORRY, we defines a default log ditectory

        TODO:
            *   Save model as numpy
            *   Predicting codes
            *   PCKh & mAP Test code
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=Global.gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)

        # model log dir control
        if log_dir is not None:
            self.writer = tf.summary.FileWriter(log_dir)
            self.log_dir = log_dir
        else:
            self.log_dir = 'logs/'
            self.writer = tf.summary.FileWriter('logs/')

        self.dataset = dataset

        # Annotations Associated
        if joints is not None:
            self.joints = joints
        else:
            self.joints = ['lane']
        self.joint_num = len(self.joints)

        #   model device control
        self.cpu = '/cpu:0'
        if cpu_only:
            self.gpu = self.cpu
        else:
            self.gpu = '/gpu:0'
            
        # Net Args
        self.stage = stage
        self.training = training
        self.base_lr = base_lr
        self.input_color = input_color
        self.in_size_w = in_size_w
        self.in_size_h = in_size_h
        
        if out_size_w is None:
            self.out_size_w = self.in_size_w / Global.pooling_scale
        else:
            self.out_size_w = out_size_w
        if out_size_h is None:
            self.out_size_h = self.in_size_h / Global.pooling_scale
        else:
            self.out_size_h = out_size_h
             
        # Ensure the consistency between pooling scale and number of decoders
        self.num_decoder = num_decoder
        assert Global.pooling_scale >= 2**self.num_decoder, 'Redundant number of decoders!'
        
        self.batch_size = batch_size
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.weight_decay = weight_decay
        
        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(base_lr,
            self.global_step, 2*self.epoch_size, 0.95, staircase=True)

        #   Inside Variable
        self.train_step = []
        self.losses = []
        self.w_summary = w_summary
        self.net_debug = False

        self.img = None
        self.gtmap = None
        
        self.summ_scalar_list = []
        self.summ_accuracy_list = []
        self.summ_image_list = []
        self.summ_histogram_list = []

        #   load model
        if pretrained_model is not None:
            self.pretrained_model = np.load(pretrained_model, encoding='latin1').item()
            print("[*]\tnumpy file loaded!")
        else:
            self.pretrained_model = None

        #   list of network parameters
        self.var_list = []


    def __build_ph(self):
        """ Building Placeholder in tensorflow session
        :return:
        """
        #   Valid & Train input
        #   input image : channel 3
        if self.input_color=='GRAY' or self.input_color=='RAW':
            self.img = tf.placeholder(tf.float32, 
                shape=[None, self.in_size_h, self.in_size_w, 1], name="img_in")
        else:
            self.img = tf.placeholder(tf.float32, 
                shape=[None, self.in_size_h, self.in_size_w, 3], name="img_in")
        #   input center map : channel 1 (downscale by 8)
        self.weight = tf.placeholder(tf.float32,
            shape=[None, self.joint_num + 1])

        #   Train input
        #   input ground truth : channel 1 (downscale by 8)
        self.gtmap = tf.placeholder(tf.float32, 
            shape=[None, self.stage, self.out_size_h, self.out_size_w, self.joint_num+1], name="gtmap")
        
        if self.num_decoder > 0:
            self.decoded_gtmap = []
            for decoder in range(self.num_decoder):
                scale = 2**(decoder+1)
                self.decoded_gtmap.append(tf.placeholder(tf.float32,
                    shape=[None, scale*self.out_size_h, scale*self.out_size_w, self.joint_num+1],
                    name='decoded_gtmap%d'%(decoder+1)))
                    
        print("place holder hdone")
        
    
    def hard_mine_loss(self, gap, epsilon=1.02):
        """ Implementing a self-adaptive weighting loss for hard mining
        """
        loss = gap**2 / 2.
        weight = tf.log(loss + epsilon)
        norm_weight = weight * tf.reduce_sum(tf.ones_like(gap)) / tf.reduce_sum(weight)
        weighted_loss = tf.multiply(weight, loss)
        return tf.reduce_sum(weighted_loss)
            

    def __build_train_op(self):
        """ Building training associates: losses & loss summary

        :return:
        """
        #   Optimizer
        with tf.name_scope('loss'):
            # Segmentation loss for stage-wise pooled heat map
            stage_loss = [tf.reduce_sum(
                tf.nn.l2_loss(self.hm[i] - self.gtmap[:, i]), name='stage%d_loss' % (i+1))
                for i in range(self.stage)]
            """
            stage_loss = [tf.reduce_sum(
                self.hard_mine_loss(self.hm[i] - self.gtmap[:, i]), name='stage%d_loss' % (i+1))
                for i in range(self.stage)]
            """
            stage_loss = tf.multiply(self.weight,
                               tf.add_n(stage_loss, name='total_stage_loss'))
            stage_loss = tf.reduce_mean(stage_loss)
            self.losses.append(stage_loss)
            
            # Segmentation loss for decoded (resolution restored) heat map
            if self.num_decoder > 0:
                decoder_loss = [tf.reduce_sum(
                    tf.nn.l2_loss(self.decoded_hm[i] - self.decoded_gtmap[i]), name='decoder%d_loss' %(i+1))
                    for i in range(self.num_decoder)]
                """
                decoder_loss = [tf.reduce_sum(
                    self.hard_mine_loss(self.decoded_hm[i] - self.decoded_gtmap[i]), name='decoder%d_loss' %(i+1))
                    for i in range(self.num_decoder)]
                """
                decoder_loss = tf.multiply(self.weight,
                                   tf.add_n(decoder_loss, name='total_decoder_loss'))
                decoder_loss = tf.reduce_mean(decoder_loss)
                self.losses.append(decoder_loss)
            
            # L2 regularization loss
            l2_var_loss = [tf.nn.l2_loss(var) for var in self.var_list]
            l2_regular = self.weight_decay * tf.add_n(l2_var_loss, name='l2_regularization')
            self.losses.append(l2_regular)
            
            self.total_loss = tf.reduce_sum(self.losses)
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.total_loss))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
            print("- LOSS & SCALAR_SUMMARY build finished!")
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                # min_learning_rate = tf.constant(0.001, name='y', dtype=tf.float32)
                # learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.99, staircase=True)
                # learning_rate = tf.where(tf.greater(learning_rate, min_learning_rate), learning_rate, min_learning_rate)

                # r_b =Global.GAUSSIAN_R

                # g_r=tf.cond(self.global_step > tf.constant(300, name='hah', dtype=tf.int32),f1,f2)
                # self.train_step.append(g_r)

                # datagen_mpii.guass_r=tf.where(tf.greater(self.global_step, 1000), Global.GAUSSIAN_r, Global.GAUSSIAN_R)
                # self.summ_scalar_list.append(tf.summary.scalar("guass_r", datagen_mpii.guass_r))

                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #   Global train
                self.train_step.append(self.optimizer.minimize(self.total_loss/self.batch_size,
                    global_step=self.global_step))
        print("- OPTIMIZER build finished!")

    
    def BuildModel(self, debug=False):
        """ Building model in tensorflow session

        :return:
        """
        #   input
        with tf.name_scope('input'):
            self.__build_ph()
        #   assertion
        assert self.img!=None and self.gtmap!=None

        if self.num_decoder > 0:
            self.hm, self.decoded_hm = self.net(self.img)
        else:
            self.hm = self.net(self.img)

        if not debug:
            #   the net
            if self.training:
                #   train op
                with tf.name_scope('train'):
                    self.__build_train_op()
                with tf.name_scope('image_summary'):
                    self.__build_monitor()
                # with tf.name_scope('accuracy'):
                #     self.__build_accuracy()
            #   initialize all variables
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            if self.training:
                #   merge all summary
                self.summ_image = tf.summary.merge(self.summ_image_list)
                self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
                # self.summ_accuracy = tf.summary.merge(self.summ_accuracy_list)
                self.summ_histogram = tf.summary.merge(self.summ_histogram_list)
        self.writer.add_graph(self.sess.graph)
        print("[*]  Model Built")

    
    def restore_sess(self, model=None):
        """ Restore session from ckpt format file

        :param model:   model path like mode
        :return:        Nothing
        """
        if model is not None:
            t = time.time()
            self.saver.restore(self.sess, model)
            print("[*]  SESS Restored!")
        else:
            print("Please input proper model path to restore!")
            raise ValueError

    
    def BuildPredict(self):
        """ builde predict tensor

        :return:
        """
        self.pred_map = tf.nn.sigmoid(self.output[:, self.stage - 1], name='sigmoid_output')
        self.pred_joints = tf.argmax(self.pred_map)

    
    def train(self):
        """ Training Progress in CPM

        :return:    Nothing to output
        """
        _epoch_count = 0
        _iter_count = 0
    
        #   datagen from Hourglass
        self.generator = self.dataset._aux_generator(self.batch_size,
                                                     stacks=self.stage, normalize=True, sample_set='train')

        self.valid_gen = self.dataset._aux_generator(self.batch_size,
                                                     stacks=self.stage, normalize=True, sample_set='valid')
                                                     
        for n in range(self.epoch):
            if(n == Global.r_change[0]):
            # if(n==1):
                datagen_mpii.guass_r=Global.GAUSSIAN_2
            elif (n == Global.r_change[1]):
                # if(n==1):
                datagen_mpii.guass_r = Global.GAUSSIAN_3
            elif (n == Global.r_change[2]):
                datagen_mpii.guass_r = Global.GAUSSIAN_4
            for m in range(self.epoch_size):
                #   datagen from hourglass
                _train_batch = next(self.generator)
                # print("[*] small batch generated!")
                for step in self.train_step:
                    if self.num_decoder > 0:
                        self.sess.run(step,
                                      feed_dict={self.img: _train_batch[0],
                                                 self.gtmap:_train_batch[1],
                                                 self.decoded_gtmap[0]:_train_batch[2][0],
                                                 self.decoded_gtmap[1]:_train_batch[2][1],
                                                 self.weight:_train_batch[3]})
                    else:
                        self.sess.run(step,
                                      feed_dict={self.img: _train_batch[0],
                                                 self.gtmap:_train_batch[1],
                                                 self.weight:_train_batch[2]})

                #   summaries
                if _iter_count % 10 == 0:
                    _test_batch = next(self.valid_gen)
                    if self.num_decoder > 0:
                        print("epoch ", _epoch_count, " iter ", _iter_count, " total loss ",
                              self.sess.run(self.total_loss,
                                            feed_dict={self.img:_test_batch[0],
                                                       self.gtmap:_test_batch[1],
                                                       self.decoded_gtmap[0]:_test_batch[2][0],
                                                       self.decoded_gtmap[1]:_test_batch[2][1],
                                                       self.weight:_test_batch[3]}))
                        #   doing the scalar summary
                        self.writer.add_summary(
                            self.sess.run(self.summ_scalar,feed_dict={self.img:_train_batch[0],
                                                            self.gtmap:_train_batch[1],
                                                            self.decoded_gtmap[0]:_train_batch[2][0],
                                                            self.decoded_gtmap[1]:_train_batch[2][1],
                                                            self.weight:_train_batch[3]}),
                            _iter_count)
                        self.writer.add_summary(
                            self.sess.run(self.summ_image, feed_dict={self.img:_test_batch[0],
                                                            self.gtmap:_test_batch[1],
                                                            self.decoded_gtmap[0]:_test_batch[2][0],
                                                            self.decoded_gtmap[1]:_test_batch[2][1],
                                                            self.weight:_test_batch[3]}),
                            _iter_count)
                        # self.writer.add_summary(
                        #     self.sess.run(self.summ_accuracy, feed_dict={self.img: _test_batch[0],
                        #                                           self.gtmap: _test_batch[1],
                        #                                           self.weight: _test_batch[2]}),
                        #     _iter_count)
                        self.writer.add_summary(
                            self.sess.run(self.summ_histogram, feed_dict={self.img: _train_batch[0],
                                                            self.gtmap:_train_batch[1],
                                                            self.decoded_gtmap[0]:_test_batch[2][0],
                                                            self.decoded_gtmap[1]:_test_batch[2][1],
                                                            self.weight:_train_batch[3]}),
                            _iter_count)
                    else:
                        print("epoch ", _epoch_count, " iter ", _iter_count, " total loss ",
                              self.sess.run(self.total_loss,
                                            feed_dict={self.img:_test_batch[0],
                                                       self.gtmap:_test_batch[1],
                                                       self.weight:_test_batch[2]}))
                        #   doing the scalar summary
                        self.writer.add_summary(
                            self.sess.run(self.summ_scalar,feed_dict={self.img:_train_batch[0],
                                                            self.gtmap:_train_batch[1],
                                                            self.weight:_train_batch[2]}),
                            _iter_count)
                        self.writer.add_summary(
                            self.sess.run(self.summ_image, feed_dict={self.img:_test_batch[0],
                                                            self.gtmap:_test_batch[1],
                                                            self.weight:_test_batch[2]}),
                            _iter_count)
                        # self.writer.add_summary(
                        #     self.sess.run(self.summ_accuracy, feed_dict={self.img: _test_batch[0],
                        #                                           self.gtmap: _test_batch[1],
                        #                                           self.weight: _test_batch[2]}),
                        #     _iter_count)
                        self.writer.add_summary(
                            self.sess.run(self.summ_histogram, feed_dict={self.img: _train_batch[0],
                                                            self.gtmap:_train_batch[1],
                                                            self.weight:_train_batch[2]}),
                            _iter_count)
                    del _test_batch

                print("iter:", _iter_count)
                _iter_count += 1
                self.writer.flush()
                del _train_batch
                
            _epoch_count += 1
            #   save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)


    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of maxlen(self.losses) position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    
    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x,u_y = self._argmax(u)
        v_x,v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))

    
    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err/num_image)

    
    def __build_accuracy(self):
        """ 
        Computes accuracy tensor
        """
        for i in range(self.joint_num):
            self.summ_accuracy_list.append(tf.summary.scalar(self.joints[i]+"_accuracy",
                                                           self._accur(self.output[self.stage-1][:, :, :, i], self.gtmap[:, self.stage-1, :, :, i], self.batch_size),
                                                           'accuracy'))
        print("- ACC_SUMMARY build finished!")

    
    def __build_monitor(self):
        """ Building image summaries

        :return:
        """
        with tf.device(self.cpu):
            # Monitor the first sample in each batch
            
            # Monitor the groundtruth map
            gt_fg = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(self.gtmap, perm=[0, 1, 4, 2, 3])[0, 0, 1:], axis=0), 0), 3)
            self.summ_image_list.append(tf.summary.image('foreground_gtmap', gt_fg, max_outputs=1))
            
            # Monitor the decoded groundtruth map for each decoder
            if self.num_decoder > 0:
                for i in range(self.num_decoder):
                    decoded_gt_fg = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(
                        self.decoded_gtmap[i], perm=[0, 3, 1, 2])[0, 1:], axis=0), 0), 3)
                    self.summ_image_list.append(tf.summary.image('foreground_decoded_gtmap%d'%(i+1), decoded_gt_fg, max_outputs=1))
            
            # Monitor the input image
            self.summ_image_list.append(tf.summary.image('image', tf.expand_dims(self.img[0], 0), max_outputs=3))
            
            # Monitor the predicted map
            for m in range(self.stage):
                pred_fg = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(
                    self.hm[m], perm=[0, 3, 1, 2])[0, 1:], axis=0), 0), 3)
                self.summ_image_list.append(tf.summary.image('foreground_hm_stage%d' %(m+1), pred_fg, max_outputs=1))

            """
            for c in range(self.joint_num):
                pred_fg = tf.expand_dims(tf.expand_dims(tf.transpose(
                    self.hm[-1], perm=[0, 3, 1, 2])[0, c+1], 0), 3)
                self.summ_image_list.append(tf.summary.image('foreground_hm_stage%d_channel%d' % (self.stage, c+1), pred_fg, max_outputs=1))
            """

            if self.num_decoder > 0:
                for t in range(self.num_decoder):
                    decoded_pred_fg = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(
                        self.decoded_hm[t], perm=[0, 3, 1, 2])[0, 1:], axis=0), 0), 3)
                    self.summ_image_list.append(tf.summary.image('foreground_hm_decoder%d' %(t+1), decoded_pred_fg, max_outputs=1))
                for c in range(self.joint_num):
                    decoded_pred_fg = tf.expand_dims(tf.expand_dims(tf.transpose(
                        self.decoded_hm[-1], perm=[0, 3, 1, 2])[0, c+1], 0), 3)
                    self.summ_image_list.append(tf.summary.image('foreground_hm_decoder%d_ch%d'%(self.num_decoder, c+1), decoded_pred_fg, max_outputs=1))
            
            if self.num_decoder > 0:
                del gt_fg, decoded_gt_fg, pred_fg, decoded_pred_fg
            else:
                del gt_fg, pred_fg
            print("- IMAGE_SUMMARY build finished!")

    
    def __TestAcc(self):
        """ Calculate Accuracy (Please use validation data)

        :return:
        """
        self.dataset.shuffle()
        assert self.dataset.idx_batches!=None
        for m in self.dataset.idx_batches:
            _train_batch = self.dataset.GenerateOneBatch()
            print("[*] small batch generated!")
            for i in range(self.joint_num):
                self.sess.run(tf.summary.scalar(i,self._accur(self.gtmap[i], self.gtmap[i], self.batch_size), 'accuracy'))

    
    def weighted_bce_loss(self):
        """ Create Weighted Loss Function
        WORK IN PROGRESS
        """
        self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtmap), name= 'cross_entropy_loss')
        e1 = tf.expand_dims(self.weight,axis = 1, name = 'expdim01')
        e2 = tf.expand_dims(e1,axis = 1, name = 'expdim02')
        e3 = tf.expand_dims(e2,axis = 1, name = 'expdim03')
        return tf.multiply(e3,self.bceloss, name = 'lossW')

    
    def net(self, image, name='CPM'):
        """ Net Structure
        Args:
            image           : Input image with n times of 8
                                size:   batch_size * in_size * in_size * sizeof(RGB)
        Return:
            stacked heatmap : Heatmap NSHWC format
                                size:   batch_size * stage_num * in_size/8 * in_size/8 * joint_num 
        """
        raise NotImplementedError
        

    #   ======= Net Component ========

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', lock=False):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs:	 Input Tensor (Data Type : NHWC)
            filters:	 Number of filters (channels)
            kernel_size: Size of kernel
            strides:     Stride
            pad:         adding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name:        Name of the block
            lock:        Lock the layer so the parameter won't be optimized
        Returns:
            outputs:	 Output Tensor (Convolved Input)
        """
        conv = tf.layers.Conv2D(filters, kernel_size, strides, padding=pad, name=name, trainable=not lock, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        outputs = conv(inputs)
        
        # Collect the layer parameters
        self.var_list.extend(conv.weights)
        
        if self.w_summary:
            with tf.device(self.cpu):
                self.summ_histogram_list.append(tf.summary.histogram(name+'_weights', conv.weights[0], collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name+'_bias', conv.weights[1], collections=['bias']))
        return outputs


    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', lock=False):
        """ Spatial Convolution (CONV2D) + Batch Norm + ReLU Activation
        Args:
            inputs:	 Input Tensor (Data Type : NHWC)
            filters:	 Number of filters (channels)
            kernel_size: Size of kernel
            strides:     Stride
            pad:         adding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name:        Name of the block
            lock:        Lock the layer so the parameter won't be optimized
        Returns:
            outputs:	 Output Tensor (Convolved Input)
        """
        conv_name = name + '_conv'
        bn_name = name + '_bn'

        conv = tf.layers.Conv2D(filters, kernel_size, strides, padding=pad, name=conv_name, trainable=not lock, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), use_bias=False)
        m1 = conv(inputs)
        
        bn = tf.layers.BatchNormalization(epsilon=1e-5, trainable=not lock, name=bn_name)
        m2 = bn(m1)
        
        outputs = tf.nn.relu(m2)
        
        # Collect the layer parameters
        self.var_list.extend(conv.weights)
        self.var_list.extend(bn.trainable_weights)
        
        if self.w_summary:
            with tf.device(self.cpu):
                self.summ_histogram_list.append(tf.summary.histogram(conv_name+'_weights', conv.weights[0], collections=['weight']))
                if lock:
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_gamma', bn.weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_beta', bn.weights[1], collections=['beta']))
                else:
                    self.summ_histogram_list.append(tf.summary.histogram(bn_name+'_gamma', bn.trainable_weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(tf.summary.histogram(bn_name+'_beta', bn.trainable_weights[1], collections=['beta']))
        return outputs

    def _atrousconv_bn_relu(self, inputs, filters, kernel_size=1, rate=1, pad='VALID', name='conv_bn_relu', lock=False):
        """ Spatial Convolution (CONV2D) + Batch Norm + ReLU Activation
        Args:
            inputs:	 Input Tensor (Data Type : NHWC)
            filters:	 Number of filters (channels)
            kernel_size: Size of kernel
            rate:     rate
            pad:         adding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name:        Name of the block
            lock:        Lock the layer so the parameter won't be optimized
        Returns:
            outputs:	 Output Tensor (Convolved Input)
        """

        with tf.variable_scope(name+"atrousconv"):
            conv_name = name + '_conv'
            bn_name = name + '_bn'
            input_channel=inputs.get_shape().as_list()[-1]

            # regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

            keneral = tf.get_variable(name+'keneral', shape=[kernel_size, kernel_size, input_channel, filters], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            # keneral =tf. get_variable()
            m1 = tf.nn.atrous_conv2d(inputs,keneral,rate=rate,padding=pad,name=conv_name)
            # conv = tf.layers.Conv2D(filters, kernel_size, strides, padding=pad, name=conv_name, trainable=not lock,
            #                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), use_bias=False)
            # m1 = conv(inputs)

            bn = tf.layers.BatchNormalization(epsilon=1e-5, trainable=not lock, name=bn_name)
            m2 = bn(m1)

            outputs = tf.nn.relu(m2)

        # Collect the layer parameters
        # self.var_list.extend(conv.weights)
        self.var_list.extend(bn.trainable_weights)

        if self.w_summary:
            with tf.device(self.cpu):
                # self.summ_histogram_list.append(
                #     tf.summary.histogram(conv_name + '_weights', conv.weights[0], collections=['weight']))
                if lock:
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_gamma', bn.weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_beta', bn.weights[1], collections=['beta']))
                else:
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_gamma', bn.trainable_weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_beta', bn.trainable_weights[1], collections=['beta']))
        return outputs
        
    
    def _deconv(self, inputs, filters, kernel_size=2, strides=2, pad='VALID', name='deconv', lock=False):
        """ Spatial Transposed Convolution (CONV2D)
        Args:
            inputs:	 Input Tensor (Data Type : NHWC)
            filters:	 Number of filters (channels)
            kernel_size: Size of kernel
            strides:     Stride
            pad:         adding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name:        Name of the block
            lock:        Lock the layer so the parameter won't be optimized
        Returns:
            outputs:	 Output Tensor (Convolved Input)
        """
        deconv = tf.layers.Conv2DTranspose(filters, kernel_size, strides, padding=pad, name=name, trainable=not lock, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        outputs = deconv(inputs)
        
        # Collect the layer parameters
        self.var_list.extend(deconv.weights)
        
        if self.w_summary:
            with tf.device(self.cpu):
                self.summ_histogram_list.append(tf.summary.histogram(name+'_weights', deconv.weights[0], collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name+'_bias', deconv.weights[1], collections=['bias']))
        return outputs
        
        
    def _deconv_bn_relu(self, inputs, filters, kernel_size=2, strides=2, pad='VALID', name='bn_relu_deconv', lock=False):
        """ Batch Norm + ReLU Activation + Spatial Transposed Convolution (CONV2D)
        Args:
            inputs:	 Input Tensor (Data Type : NHWC)
            filters:	 Number of filters (channels)
            kernel_size: Size of kernel
            strides:     Stride
            pad:         adding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name:        Name of the block
            lock:        Lock the layer so the parameter won't be optimized
        Returns:
            outputs:	 Output Tensor (Convolved Input)
        """
        deconv_name = name + '_deconv'
        bn_name = name + '_bn'        
        
        deconv = tf.layers.Conv2DTranspose(filters, kernel_size, strides, padding=pad, name=deconv_name, trainable=not lock, 
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), use_bias=False)
        m1 = deconv(inputs)
        
        bn = tf.layers.BatchNormalization(epsilon=1e-5, trainable=not lock, name=bn_name)
        m2 = bn(m1)
        
        outputs = tf.nn.relu(m2)
         
        # Collect the layer parameters
        self.var_list.extend(deconv.weights)
        self.var_list.extend(bn.trainable_weights)
        
        if self.w_summary:
            with tf.device(self.cpu):
                self.summ_histogram_list.append(tf.summary.histogram(deconv_name+'_weights', deconv.weights[0], collections=['weight']))
                if lock:
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_gamma', bn.weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(bn_name + '_beta', bn.weights[1], collections=['beta']))
                else:
                    self.summ_histogram_list.append(tf.summary.histogram(bn_name+'_gamma', bn.trainable_weights[0], collections=['gamma']))
                    self.summ_histogram_list.append(tf.summary.histogram(bn_name+'_beta', bn.trainable_weights[1], collections=['beta']))
        return outputs
    
    
    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        with tf.variable_scope(name):
            with tf.variable_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv1')
            with tf.variable_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv2')
            with tf.variable_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv3')
            return conv_3
                
    
    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv_sk')
                return conv				
    
    
    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name):
            convb = self._conv_block(inputs, numOut, name='_conv_bl')
            skipl = self._skip_layer(inputs, numOut, name='_conv_sk')
            if self.net_debug:
                return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
            else:
                return tf.add_n([convb, skipl], name = 'res_block')

