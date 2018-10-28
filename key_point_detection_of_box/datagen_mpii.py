# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Wed Jul 12 15:53:44 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
        This python code creates a Stacked Hourglass Model
        (Credits : A.Newell et al.)
        (Paper : https://arxiv.org/abs/1603.06937)

        Code translated from 'anewell' github
        Torch7(LUA) --> TensorFlow(PYTHON)
        (Code : https://github.com/anewell/pose-hg-train)

        Modification are made and explained in the report
        Goal : Achieve Real Time detection (Webcam)
        ----- Modifications made to obtain faster results (trade off speed/accuracy)

        This work is free of use, please cite the author if you use it!
========================================================================
P.S.:
    This is a modified version of the origin HG model
    It is free to scale up and down and more flexible for experiments
    Net Structure might be different from the master branch

TODO:
    Data generater should be able to generate bounding box
    to compensate the error during multi-persion sample training
    This would help net get normalized and balanced very quickly

    This Arch can be use in Densepose complementation(need to be tested)

    mpsk	2018-03-02
"""
import numpy as np
import Global
import os
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform, exposure
import scipy.misc as scm
from PIL import Image

guass_r = Global.GAUSSIAN_1

class DataGenerator(object):
    """ DataGenerator Class : To generate Train, Validatidation and Test sets
    for the Deep Human Pose Estimation Model
    Formalized DATA:
        Inputs:
            Inputs have a shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3)
        Outputs:
            Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: self.out_size) X (Width: self.out_size) X (OutputDimendion: 16)
    Joints:
        We use the MPII convention on joints numbering
        List of joints:
            00 - Right Ankle
            01 - Right Knee
            02 - Right Hip
            03 - Left Hip
            04 - Left Knee
            05 - Left Ankle
            06 - Pelvis (Not present in other dataset ex : LSP)
            07 - Thorax (Not present in other dataset ex : LSP)
            08 - Neck
            09 - Top Head
            10 - Right Wrist
            11 - Right Elbow
            12 - Right Shoulder
            13 - Left Shoulder
            14 - Left Elbow
            15 - Left Wrist
    # TODO : Modify selection of joints for Training

    How to generate Dataset:
        Create a TEXT file with the following structure:
            image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
            [LETTER]:
                One image can contain multiple person. To use the same image
                finish the image with a CAPITAL letter [A,B,C...] for
                first/second/third... person in the image
             joints :
                Sequence of x_p y_p (p being the p-joint)
                /!\ In case of missing values use -1

    The Generator will read the TEXT file to create a dictionnary
    Then 2 options are available for training:
        Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
        Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration)
    """

    def __init__(self,
                 input_color='GRAY',
                 joints_name=None,
                 img_train_dir=None,
                 img_valid_dir=None,
                 label_train_dir=None,
                 label_valid_dir=None,
                 train_data_file=None,
                 remove_joints=None,
                 in_size_h=720,
                 in_size_w=1280,
                 ann_size_h=720,
                 ann_size_w=1280,
                 decode_scale=[2],
                 Gaussian_r=5,
                 out_size_h=None,
                 out_size_w=None,
                 aug_data=True):
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        self.input_color = input_color
        if joints_name == None:
            self.joints_list = ['side', 'block', 'pass']
        else:
            self.joints_list = joints_name
        self.toReduce = False
        if remove_joints is not None:
            self.toReduce = True
            self.weightJ = remove_joints
            
        # Map from label to joints name
        self.joints_map = {'1': '1',
                           '2': '2',
                           '3': '3',
                           '4': '4',
                           '5': '5',
                           '6': '6',
                           '7': '7'}

        self.ann_size_h = ann_size_h
        self.ann_size_w = ann_size_w
        
        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        
        if out_size_h is None:
            self.out_size_h = self.in_size_h // Global.pooling_scale
        else:
            self.out_size_h = out_size_h
        if out_size_w is None:
            self.out_size_w = self.in_size_w // Global.pooling_scale
        else:
            self.out_size_w = out_size_w
            
        self.decode_scale = decode_scale
        if len(self.decode_scale) > 0:
            self.decode_size_w = []
            self.decode_size_h = []
            for scale in self.decode_scale:
                self.decode_size_w.append(self.in_size_w // scale)
                self.decode_size_h.append(self.in_size_h // scale)    
            
        self.Gaussian_r = Gaussian_r
            
        self.joints_num = len(self.joints_list)
        self.aug_data = aug_data

        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.img_train_dir = img_train_dir
        self.img_valid_dir = img_valid_dir
        self.label_train_dir = label_train_dir
        self.label_valid_dir = label_valid_dir
        
        self.images_train = os.listdir(img_train_dir)
        self.images_valid = os.listdir(img_valid_dir)

    # --------------------Generator Initialization Methods ---------------------

    def _reduce_joints(self, joints):
        """ Select Joints of interest from self.weightJ
        """
        j = []
        for i in range(len(self.weightJ)):
            if self.weightJ[i] == 1:
                j.append(joints[2 * i])
                j.append(joints[2 * i + 1])
        return j


    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []
        self.data_train_dict = {}
        for filename in os.listdir(self.label_train_dir):
            # print(filename.split('.')[0])
            filename = filename.split('.')[0]
            input_file = open(self.label_train_dir + '/' + str(filename) + '.txt', 'r')
            joint = []
            position = []
            for line in input_file:
                line = line.strip()
                line = line.split(' ')
                
                # Select part of joints from the whole set to train
                if line[-4] == '0':
                    continue

                joint_name = self.joints_map[line[-4]]
                # joint_name = '1'


                # Collect all the keypoint categories into a uniform category

                if joint_name in self.joints_list:
                    joint.append(self.joints_list.index(joint_name)+1)
                    # especially for describing point position
                    one_position = line[1:3]
                    position.append(one_position)
                    
            w = [1]
                
            # Normalize the position to deal with image scaling
            self.data_train_dict[filename] = {'joint': joint, 'position': position, 'weights': w}
            self.train_table.append(filename)


    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)


    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
                return False
        return True


    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set				: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file


    def _create_sets(self, validation_rate=0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in (0,1), don't waste time use 0.1)
        """
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = self.train_table[sample - valid_sample:]
        # self.train_set = self.train_table
        # self.valid_set = self.valid_table
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')
        

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    # ---------------------------- Generating Methods --------------------------

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    
    def _generate_hm(self, joint, position, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            joints			: Array of Joints
        """
        num_joints = len(joint)
        hm = np.zeros((self.out_size_h, self.out_size_w, len(self.joints_list)+1), dtype=np.float32)
        if len(self.decode_scale) > 0:
            decoded_hm = []
            for i, scale in enumerate(self.decode_scale):
                decoded_hm.append(np.zeros((self.decode_size_h[i], self.decode_size_w[i], len(self.joints_list)+1), dtype=np.float32))

        for i in range(num_joints):
            # Especially for describing point position.
            # Record on the heat map.
            hm_h = int(position[i][1]) * self.out_size_h // self.ann_size_h
            hm_w = int(position[i][0]) * self.out_size_w // self.ann_size_w
            if( hm_h > self.out_size_h-1):
                hm_h = hm_h-1
            if (hm_w > self.out_size_w - 1):
                hm_w = hm_w - 1
            hm[hm_h, hm_w, int(joint[i])] = 255.
            if len(self.decode_scale) > 0:
                # Record on the decoded heat map
                for j, scale in enumerate(self.decode_scale):

                    decode_hm_h = int(position[i][1]) * self.decode_size_h[j] // self.ann_size_h
                    decode_hm_w = int(position[i][0]) * self.decode_size_w[j] // self.ann_size_w

                    if(decode_hm_h>self.decode_size_h[j]-1):
                        decode_hm_h=self.decode_size_h[j]-1
                    if (decode_hm_w > self.decode_size_w[j] - 1):
                        decode_hm_w = self.decode_size_w[j] - 1

                    decoded_hm[j][decode_hm_h, decode_hm_w, int(joint[i])] = 255.

        #----------------------------------------
        # hm = cv2.GaussianBlur(hm, (self.Gaussian_r, self.Gaussian_r), 0)
        hm = cv2.GaussianBlur(hm, (guass_r, guass_r), 0)

        # ----------------------------------------
        if len(self.decode_scale) > 0:
            for i, m in enumerate(decoded_hm):
                decoded_hm[i] = cv2.GaussianBlur(m, (guass_r, guass_r), 0)
        
        for i in range(1, self.joints_num+1):
            if np.max(hm[:, :, i]) > 1e-10:
                hm[:, :, i] = hm[:, :, i] / np.max(hm[:, :, i])
                if len(self.decode_scale) > 0:
                    for j, m in enumerate(decoded_hm):
                        decoded_hm[j][:, :, i] = decoded_hm[j][:, :, i] / np.max(decoded_hm[j][:, :, i])
                
        # Generate supervising information for background
        anti_hm = np.add.reduce(hm[:, :, 1:], axis=-1)
        if np.max(anti_hm) > 1e-10:
            anti_hm = anti_hm / np.max(anti_hm)
        hm[:, :, 0] = 1. - anti_hm
        if len(self.decode_scale) > 0:
            for i, m in enumerate(decoded_hm):
                anti_hm = np.add.reduce(decoded_hm[i][:, :, 1:], axis=-1)
                if np.max(anti_hm) > 1e-10:
                    anti_hm = anti_hm / np.max(anti_hm)
                decoded_hm[i][:, :, 0] = 1. - anti_hm
                
        if len(self.decode_scale) > 0:
            return hm, decoded_hm
        else:
            return hm
        

    def _augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        # Decide whether to perform rotation, not proper for traffic scene
        #
        # if random.choice([0, 1]):
        #     r_angle = np.random.randint(-1 * max_rotation, max_rotation)
        #     img = transform.rotate(img, r_angle, preserve_range=True)
        #     hm = transform.rotate(hm, r_angle)

        # Decide whether to perform brightness adjustment
        if random.choice([0, 1]):
            # Brightness enhancement or decay
            if random.choice([0, 1]):
                gamma = np.random.uniform(0.5, 1.)
            else:
                gamma = np.random.uniform(1., 2.)
            img = exposure.adjust_gamma(img, gamma=gamma)
        return img, hm

    # ----------------------- Batch Generator ----------------------------------

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            if self.input_color=='GRAY' or self.input_color=='RAW':
                train_img = np.zeros((batch_size, self.in_size_h, self.in_size_w, 1), dtype=np.float32)
            else:
                train_img = np.zeros((batch_size, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, self.out_size_h, self.out_size_w, len(self.joints_list)+1),
                                   dtype=np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list)+1), np.float32)
            
            if len(self.decode_scale) > 0:
                train_decoded_gtmap = []
                for j, scale in enumerate(self.decode_scale):
                    train_decoded_gtmap.append(np.zeros((batch_size, self.decode_size_h[j], self.decode_size_w[j], len(self.joints_list)+1), dtype=np.float32))
            
            i = 0
            while i < batch_size:
                # try:
                if sample_set == 'train':
                    name = random.choice(self.train_set)

                elif sample_set == 'valid':
                    name = random.choice(self.valid_set)

                # print(name)
                # img = self.open_img_train(name)
                img = self.open_img_train(name, color=self.input_color)
                joint = self.data_train_dict[name]['joint']
                position = self.data_train_dict[name]['position']
                weight = self.data_train_dict[name]['weights']
                if len(self.decode_scale) > 0:
                    hm, decoded_hm = self._generate_hm(joint, position, weight)
                else:
                    hm = self._generate_hm(joint, position, weight)
 
                train_weights[i][:len(self.joints_list)] = weight

                if self.input_color != 'RAW':
                    img = img.astype(np.uint8)
                    img = scm.imresize(img, (self.in_size_h, self.in_size_w))
                    # expand the channel dimension for gray images
                    if len(img.shape)<3:
                        img = np.expand_dims(img, axis=-1)
                    if self.aug_data:
                        img, hm = self._augment(img, hm)
                    if normalize:
                        train_img[i] = img.astype(np.float32) / 255
                    else:
                        train_img[i] = img.astype(np.float32)
                else:
                    img = np.expand_dims(img, axis=-1)
                    train_img[i] = img
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                
                train_gtmap[i] = hm
                if len(self.decode_scale) > 0:
                    for j, scale in enumerate(self.decode_scale):
                        train_decoded_gtmap[j][i] = np.expand_dims(decoded_hm[j], axis=0)
                i = i + 1

                # print(np.max(train_img), np.min(train_img))
            # print (train_decoded_gtmap)
            if len(self.decode_scale) > 0:
                yield train_img, train_gtmap, train_decoded_gtmap, train_weights
            else:
                yield train_img, train_gtmap, train_weights

    def generator(self, batchSize=16, stacks=4, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img_train(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        if color == 'RAW':
            bytes = open(os.path.join(self.img_train_dir, name+'.raw'), 'rb').read()
            img = Image.frombytes('F', (self.in_size_w, self.in_size_h), bytes, 'raw')
            img = np.array(img, dtype=np.float32)
            return img
        else:
            img = cv2.imread(os.path.join(self.img_train_dir, name + '.jpg'))

        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR/GRAY. If you need another mode do it yourself :p')

    def open_img_valid(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.img_valid_dir, name + '.jpg'))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def open_heatmap_train(self, name, color='GRAY'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.label_train_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def open_heatmap_valid(self, name, color='GRAY'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.label_valid_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._create_train_table()
        self._create_sets()
        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'],
                                        self.data_dict[self.train_set[i]]['joints'], boxp=0.0)
            new_j = self._relative_joints(box, padd, self.data_dict[self.train_set[i]]['joints'], to_size=self.in_size)
            rhm = self._generate_hm(self.in_size, self.in_size, new_j, self.in_size, w)
            rimg = self._crop_img(img, padd, box)
            # See Error in self._generator
            # rimg = cv2.resize(rimg, (self.in_size,self.in_size))
            rimg = scm.imresize(rimg, (self.in_size, self.in_size))
            # rhm = np.zeros((self.in_size,self.in_size,16))
            # for i in range(16):
            #	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (self.in_size,self.in_size))
            grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2))
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

    # ------------------------------- PCK METHODS-------------------------------
    def pck_ready(self, idlh=3, idrs=12, testSet=None):
        """ Creates a list with all PCK ready samples
        (PCK: Percentage of Correct Keypoints)
        """
        id_lhip = idlh
        id_rsho = idrs
        self.total_joints = 0
        self.pck_samples = []
        for s in self.data_dict.keys():
            if testSet == None:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][
                    id_rsho] == 1 and s in testSet:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

    def getSample(self, sample=None):
        """ Returns information of a sample
        Args:
            sample : (str) Name of the sample
        Returns:
            img: RGB Image
            new_j: Resized Joints
            w: Weights of Joints
            joint_full: Raw Joints
            max_l: Maximum Size of Input Image
        """
        if sample != None:
            joints = self.data_dict[sample]['joints']
            box = self.data_dict[sample]['box']
            w = self.data_dict[sample]['weights']
            img = self.open_img(sample)
            padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
            new_j = self._relative_joints(cbox, padd, joints, to_size=self.in_size)
            joint_full = np.copy(joints)
            max_l = max(cbox[2], cbox[3])
            joint_full = joint_full + [padd[1][0], padd[0][0]]
            joint_full = joint_full - [cbox[0] - max_l // 2, cbox[1] - max_l // 2]
            img = self._crop_img(img, padd, cbox)
            img = img.astype(np.uint8)
            img = scm.imresize(img, (self.in_size, self.in_size))
            return img, new_j, w, joint_full, max_l
        else:
            print('Specify a sample name')


if __name__ == '__main__':
    #   module testing code
    INPUT_SIZE = 368
    IMG_ROOT = "/Documents/mpii"
    training_txt_file = "dataset.txt"
    joint_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head',
                  'r_wrist', 'r_elbow', 'r_shoulder', 'l_sho    ulder', 'l_elbow', 'l_wrist']
    gen = DataGenerator(joint_list, IMG_ROOT, training_txt_file, remove_joints=None, in_size=INPUT_SIZE)
