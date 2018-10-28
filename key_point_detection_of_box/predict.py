#coding: utf-8

"""
    ICHIGO PROJ

    2018-03-26
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology
"""
import cv2
import numpy as np
import os
import Global
import CPM
from skimage import transform, io, color, exposure
import datetime
import time
import scipy.misc as scm

BATCH_SIZE = 64

in_size_h = 224
in_size_w = 172
# joints_list = ['dashed_line', 'solid_line', 'stop_line', 'straight', 'left', 'right', 'back', 'straight_right', 'left_back',
    # 'straight_left', 'straight_back']
joints_list=['1']
joint_num = len(joints_list)
input_color = 'RGB'

model_path = 'logs/20180828_1541/model.ckpt-29'
video_name = '163448AA'
raw_root = '/home/lechatelia/Desktop/Codes/TOF/images/' #"/home/robot/Dataset/lane_detection/select/" # "test_set/images/"
#vis_root = "/home/robot/Project/CPM/data/test_results/"
hm_root = "/home/lechatelia/Desktop/Codes/TOF/results/"
if not os.path.exists(hm_root):
    os.mkdir(hm_root)

#hm_root = '/home/lechatelia/Desktop/CPM-boundary-data/' + video_name + '/stage6_htmap_0531/'
pre_root = '/home/lechatelia/Desktop/CPM-boundary-data/predict/predict_pictures/'
video_root = '/home/lechatelia/Dataset/Raw/lane/videos/'

video_root = os.path.join('/home/robot/Experiments/lane_CPM/test_set', video_name)
pic_sav_root = os.path.join(video_root, 'images')


def resize_to_imgsz(hm, img):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :return:
    """
#    assert len(hm.shape) == 4 and len(img.shape) == 4
    ret_map = np.zeros((hm.shape[0], img.shape[1], img.shape[2], hm.shape[-1]), np.float32)
    for n in range(hm.shape[0]):
        for c in range(hm.shape[-1]):
            ret_map[n,:,:,c] = transform.resize(hm[n,:,:,c], img.shape[1: 3])
    return ret_map

def joints_plot_image(joints, img, radius=3, thickness=2):
    """ Plot the joints on image

    :param joints:      (np.array)Assuming input of shape (joint_num, dim)
    :param img:         (image)Assuming
    :param radius:      (int)Radius
    :param thickness:   (int)Thickness
    :return:        RGB image
    """
    #assert len(joints.shape)==3 and len(img.shape)==4
    colors = [(241,242,224), (196,203,128), (136,150,0), (64,77,0), 
            (201,230,200), (132,199,129), (71,160,67), (32,94,27),
            (130,224,255), (7,193,255), (0,160,255), (0,111,255),
            (220,216,207), (174,164,144), (139,125,96), (100,90,69),
            (252,229,179), (247,195,79), (229,155,3), (155,87,1),
            (231,190,225), (200,104,186), (176,39,156), (162,31,123),
            (210,205,255), (115,115,229), (80,83,239), (40,40,198)]
    ret = np.zeros(img.shape, np.uint8)
    for num in range(joints.shape[0]):
        for jnum in range(joints.shape[1]):
            print(tuple(joints[num, jnum].astype(int)))
            ret[num] = cv2.circle(img[num], (int(joints[num,jnum,1]), int(joints[num,jnum,0])),
                                  radius=radius, color=colors[jnum], thickness=thickness)
    return ret

def my_find(n, i, hm, thresh):
    index = []
    for height in range(hm.shape[1]):
        for width in range(hm.shape[2]):
            if hm[n, height, width, i] > thresh:
                index.append(height)
                index.append(width)
    return index

def joints_pred_numpy(hm, img, coord='img', thresh=0.2):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :param  coord:  project to original image or not
    :param  thresh: Threshold to limit small respond
    :return:
    """
    colors = [(241,242,224), (196,203,128), (136,150,0), (64,77,0),
            (201,230,200), (132,199,129), (71,160,67), (32,94,27),
            (130,224,255), (7,193,255), (0,160,255), (0,111,255),
            (220,216,207), (174,164,144), (139,125,96), (100,90,69),
            (252,229,179), (247,195,79), (229,155,3), (155,87,1),
            (231,190,225), (200,104,186), (176,39,156), (162,31,123),
            (210,205,255), (115,115,229), (80,83,239), (40,40,198)]
    ret = np.zeros(img.shape, np.uint8)
    # assert len(hm.shape) == 4 and len(img.shape) == 4
    joints = -1 * np.ones(shape=(hm.shape[0], joint_num, 2)) # b, 16, 2
    weight = np.zeros(shape=(hm.shape[0], joint_num)) # b 16
    for n in range(hm.shape[0]): # b
        IMG = np.zeros((int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), int(img.shape[3]), int(joint_num)), np.uint8)
        for i in range(joint_num): # 16
            ret[n] = img[n]
            # np.maximum(hm[n, :, :, i], 0)
            pix_max = np.max(hm[n, :, :, i])
            pix_min = np.min(hm[n, :, :, i])
            # index = my_find(n, i, hm, 0)
            # for dex in range(len(index)//2):
            #     if hm[n, index[2*dex], index[2*dex+1], i] > thresh:
            #             ret[n][index[2*dex]][index[2*dex+1]] = ret[n][index[2*dex]][index[2*dex+1]]*(1 + 1/(pix_max-pix_min)*hm[n, index[2*dex], index[2*dex+1], i])
            aver_intensity = np.sum(ret[n, :, :, i])/(ret.shape[1]*ret.shape[2])
            if aver_intensity<80:
                jiaozheng = 2
            else:
                jiaozheng = 1
            # xuejian = int(aver_intensity/255*5)
            # k = (5 - 1)/(1 - 255)
            # b = 1 - 255*k
            # tisheng = int(k*255/aver_intensity + b) - 1
            # if tisheng > xuejian*int(255/aver_intensity) - 1:
            #     tisheng = xuejian*int(255/aver_intensity) - 1
            # print("xuejian",xuejian)
            # print("tisheng",tisheng)
            for col in range(hm.shape[1]):
                for row in range(hm.shape[2]):
                    ret[n][col][row] = ret[n][col][row] / 3 * (1 + 2*jiaozheng / (pix_max - pix_min) * hm[n, col, row, i])
            IMG[n, :, :, :, i] = ret[n, :, :, :]
            # ret[n] = cv2.circle(img[n], (int(index[2*dex+1]*img.shape[2]/hm.shape[2]), int(index[2*dex]*img.shape[1]/hm.shape[1])),
                    #                      radius=1, color=colors[i], thickness=0)
            # index = np.unravel_index(hm[n, :, :, i].argmax(), hm.shape[1: 3])
            # print(index)
            # print(index) # (i_0, i_1)
            # if hm[n, index[0], index[1], i] > thresh:
            #     if coord == 'hm':
            #         joints[n, i] = np.array(index)
            #     elif coord == 'img':
            #         joints[n, i] = np.array(index) * (img.shape[1], img.shape[2]) / hm.shape[1: 3]
            #     weight[n, i] = 1
    return joints, weight, ret, IMG


def predict(img_list,
            model_path=None,
            thresh=0.05,
            is_name=False,
            cpu_only=True,
            model=None,
            id=0,
            num=0,
            debug=False):
    """ predict API
    You can input any size of image in this function. and the joint result is remaped to 
    the origin image according to each image's scale
    Just feel free to scale up n down :P

    Just be aware of the `is_name` param: this is to determine if
    
    :param img_list:    list of img (numpy array) !EVEN ONE PIC VAL ALSO NEED INPUT AS LIST!
    :param model_path:  path to load the model
    :param thresh:      threshold value (ignore some small peak)
    :param is_name:     define whether the input is name_list or numpy_list
    :param cpu_only:    CPU only mode or GPU accelerate mode
    :param model:       preload model to do the predict
    :return :
    """
    #   Assertion to check input format
    assert img_list != None and len(img_list) >= 1
    if model == None and model_path == None:
        print('[!]  Error!  A mo(len(_img_list), pred_map.shape[-1]-1, 2)del or a model path must be given!')
        raise ValueError
    if is_name:
        assert type(img_list[0]) == str
    if is_name == True:
        _img_list = []
    else:
        _img_list = img_list
    input_list = []
    for idx in range(len(img_list)):
        try:
            if is_name == True:
                t_img = io.imread(raw_root + img_list[idx])
                # Test the model in various lighting conditions
                """
                if np.random.randint(2):
                    gamma = np.random.uniform(0.5, 1.)
                else:
                    gamma = np.random.uniform(1., 2.)
                t_img = exposure.adjust_gamma(t_img, gamma=gamma)
                """
                
                if input_color=='GRAY':
                    t_img = color.rgb2gray(t_img)
                if t_img is None:
                    raise IOError
                _img_list.append(t_img)
            else:
                t_img = _img_list[idx]
            input_list.append(cv2.resize(t_img, (in_size_w, in_size_h)))
        except:
            print('[!]  Error!  Failed to load image of index_' + str(idx))
    #   convert list to numpy array
    """
    _input = np.array(input_list)
    if len(_input.shape)==3:
        _input = np.expand_dims(_input, axis=-1)
    """
    # Detach the whole list into smaller batches to fit the GPU memory limit
    batch_input = []
    batches = int(len(input_list) / BATCH_SIZE)
    for batch in range(batches):
        _input = np.array(input_list[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE])
        if len(_input.shape) == 3:
            _input = np.expand_dims(_input, axis=-1)
        batch_input.append(_input)
    if len(input_list) > batches*BATCH_SIZE:
        _input = np.array(input_list[batches*BATCH_SIZE:])
        if len(_input.shape) == 3:
            _input = np.expand_dims(_input, axis=-1)
        batch_input.append(_input)
        
    # Load the model and weights
    if model == None:
        model = \
            CPM.CPM(pretrained_model=None,
                    joints=joints_list,
                    input_color=input_color,
                    in_size_h=in_size_h,
                    in_size_w=in_size_w,
                    cpu_only=False,
                    training=False)
        model.BuildModel()
    if model_path is not None:
        model.restore_sess(model_path)

    #   get the last stage's result
    starttime = datetime.datetime.now().microsecond

    batch_output = []
    for _input in batch_input:
        _input = _input / 255.
        pred_map = model.sess.run(model.output, feed_dict={model.img: _input})[-1]
        batch_output.append(pred_map)
    """
    _input = _input / 255.
    pred_map = model.sess.run(model.output, feed_dict={model.img: _input})[-1]
    """
    endtime = datetime.datetime.now().microsecond

    print('********************')
    print(endtime - starttime)

    pred_map = np.concatenate(batch_output, axis=0)
    # pred_map 4 46 46 17 <- 4 6 46 46 17
    #pred_map1 = model.sess.run(model.output, feed_dict={model.img: _input / 255.0})
    print(pred_map.shape)
    for idx, element in enumerate(img_list):
        r_pred_map = resize_to_imgsz(np.expand_dims(pred_map[idx], 0),
                                     np.expand_dims(_img_list[idx], 0))
        print (r_pred_map.shape)
        if debug:
            # Save heat maps of lanes and markings separately, used in testing the model jointly 
            # trained by lanes and markings.
            # Lanes: channel 1~3
            laneName = os.path.join(hm_root, '_'.join(element.split('.')[:-1])+'_lane'+'.jpg')
            laneMap = np.add.reduce(r_pred_map[0, :, :, 1:4], axis=-1)
            if np.max(laneMap) > 1e-10:
                laneMap = laneMap / np.max(laneMap)
            io.imsave(laneName, (laneMap*255).astype(np.uint8))
            # Markings: channel 4~11
            """
            markName = os.path.join(hm_root, '_'.join(element.split('.')[:-1])+'_marking'+'.jpg')
            markMap = np.add.reduce(r_pred_map[0, :, :, 4:], axis=-1)
            if np.max(markMap) > 1e-10:
                markMap = markMap / np.max(markMap)
            io.imsave(markName, (markMap*255).astype(np.uint8))
            """
            
            """
            # Save predicted heat map of background and each joint
            for channel in range(r_pred_map.shape[-1]):
                saveName = os.path.join(hm_root, '_'.join(element.split('.')[:-1])+'_channel_%d'%(channel)+'.jpg')
                saveMap = r_pred_map[0, :, :, channel]
                io.imsave(saveName, (saveMap * 255).astype(np.uint8))
            """


def imgs2video(imgs_dir, save_dir):
    fps = 30  # 保存视频的FPS，可以适当调整
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(save_dir+'lane_pred_video.avi', fourcc, fps, (1280, 720))  # 最后一个是保存图片的尺寸

    for i in range(len(os.listdir(imgs_dir))):
        img = '%04d_lane.jpg' %(i)
        frame = cv2.imread(os.path.join(imgs_dir, img))
        if frame is None:
            continue
        else:
            print (img)
            videoWriter.write(frame)
    videoWriter.release()


if __name__=='__main__':
    """ Demo of Using the model API
    """
    
    # import argparse
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(description='Demo of using the model API')
    # parser.add_argument('--mode', required=False, default=1,
    #                     metavar='control mode',
    #                     help='0 for video2frame, 1 for prediction, 2 for frame2video')
    # args = parser.parse_args()
    # control_mod = int(args.mode)
    control_mod = 1
    if control_mod == 0:###将视频保存为图片帧
        video_full_path = os.path.join('/home/robot/Experiments/lane_CPM', video_name+'.MP4')
        cv2.namedWindow('video')
        cap = cv2.VideoCapture(video_full_path)
        frame_count = 0
        success = True
        while(1):
            print(frame_count)
            success, frame = cap.read()
            real_frame = cv2.resize(frame, (1280, 720))
            if not os.path.exists(pic_sav_root):
                if not os.path.exists(video_root):
                    os.mkdir(video_root)
                os.mkdir(pic_sav_root)
            img_file_name = os.path.join(pic_sav_root, '%04d.jpg'%(frame_count))
            cv2.imwrite(img_file_name, real_frame)
            frame_count = frame_count+1
    elif control_mod == 1:###预测

        num = -1
        cishu = 0
        img_names = []
        for root, dirs, files in os.walk(raw_root):
            img_names = files

        # j = predict(img_names, '/home/Fdisk/CPM/model/tf17model/model.ckpt-49', num=num,
        #             debug=True, is_name=True)
        """
        j = predict(img_names, '/home/young/Experiments/CPM/model/2448model/model.ckpt-49', num=num,
                    debug=True, is_name=True)
        """
        predict(img_names, model_path, num=num, debug=True, is_name=True)



    elif control_mod == 2:
        imgs2video(hm_root, "test_set/154002AA/")
