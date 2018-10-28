#	Global variable goes here
#	first to define the dataset root
from datetime import datetime
now = datetime.now()

IMG_TRAIN_ROOT = "/home/lechatelia/Desktop/Codes/TOF/images"
IMG_VALID_ROOT = "/home/lechatelia/Desktop/Codes/TOF/images"
LABEL_TRAIN_ROOT = "/home/lechatelia/Desktop/Codes/TOF/annotations"
LABEL_VALID_ROOT = "/home/lechatelia/Desktop/Codes/TOF/annotations"

# INPUT_SIZE_W = 896
# INPUT_SIZE_H = 688
INPUT_SIZE_W = 112
INPUT_SIZE_H = 88
ANN_SIZE_W = 896
ANN_SIZE_H = 688
GAUSSIAN_1 = 33
GAUSSIAN_R=GAUSSIAN_1
GAUSSIAN_2 = 23
GAUSSIAN_3 = 13
GAUSSIAN_4 = 9
# r_change=[35,55,75,90]
r_change=[35,60,80]#,105]
DECODE_SCALE = [2,1]

AUG_DATA = False
# AUG_DATA = True
base_lr = 1e-4
# epoch = 100
epoch = 100
epoch_size = 1000
batch_size = 8
gpu_memory_fraction = 1.0

# Set the pooling scale to 4 when using deconv. for decoding
pooling_scale = 4

LOGDIR = "log1024/%d%02d%02d_%02d%02d/" %(
    now.year, now.month, now.day, now.hour, now.minute)
 
joint_list = ['1', '2', '3', '4', '5', '6', '7']
# joint_list = ['1']
# input_color = 'RGB''RGB'
input_color = 'GRAY'

# Set the model architecture, note that the number of decoders 
# be consistent with decode scale above, and the number of stages
# be consistent with the defined model in CPM.py .
STAGE = 5
NUM_DECODER = 2  #更改decode 数量步骤
#1, change NUM_DECODER
#2, change DECODE_SCALE as the output of the decode stage relative to the input size
#3, change the feed function as
# self.decoded_gtmap[0]: _test_batch[2][0],
# self.decoded_gtmap[1]: _test_batch[2][1],

