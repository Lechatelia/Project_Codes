# import numpy as np
# import cv2
# canvas = np.zeros((300, 300, 3), dtype="uint8")
# #画绿线
# green = (255,255,255)
# position = [['17', '45', '47', '79'], ['3', '18', '2', '78', '13', '48', '56', '26'], ['26', '46', '78', '47'], ['51', '47', '75', '46']]
#
# for i in range(4):
#     num_positions = len(position[i])
#     for j in range(num_positions // 4):
#         cv2.line(canvas, (int(position[i][4 * j]), int(position[i][4 * j + 1])), (int(position[i][4 * j + 2]), int(position[i][4 * j + 3])), green, 2)
# # for i in range(4):
# #     for j in range(len(position[i])//4):
#
# cv2.imshow("Canvas",canvas)
# cv2.waitKey(0)

import numpy as np
import cv2
# gray = np.zeros((300, 300, 4), dtype="uint8")
# bgr = cv2.cvtColor(gray[:, :, 1], cv2.COLOR_GRAY2BGR)
# #画绿线
# green = (255,255,255)
# position = [['17', '45', '47', '79'], ['3', '18', '2', '78', '13', '48', '56', '26'], ['26', '46', '78', '47'], ['51', '47', '75', '46']]
#
# for i in range(4):
#     num_positions = len(position[i])
#     for j in range(num_positions // 4):
#         cv2.line(bgr, (int(position[i][4 * j]), int(position[i][4 * j + 1])), (int(position[i][4 * j + 2]), int(position[i][4 * j + 3])), green, 2)
# # for i in range(4):
# #     for j in range(len(position[i])//4):
# gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Canvas", gray)
# cv2.waitKey(0)

# joint = ['0', '0', '0', '1', '1', '1', '2', '2', '2', '2', '2']
# position = [['14', '472', '277', '405', '336', '385', '387', '367', '410', '355'], ['623', '351', '679', '362', '709', '369', '735', '373', '756', '379', '778', '385', '793', '390'], ['1009', '439', '1265', '494'], ['428', '355', '465', '355'], ['489', '378', '537', '377'], ['562', '361', '567', '368', '605', '367'], ['410', '355', '430', '355'], ['466', '355', '490', '380'], ['537', '378', '560', '359'], ['606', '368', '624', '351'], ['794', '391', '1010', '440']]
#
# pic_folder = "/home/Fdisk/CPM/2448data/pic/"
# filename = "00178"
# folder = '/home/Fdisk/CPM/2448data/annotation'
# input_file = open(folder + '/' + str(filename) + '.txt', 'r')
# hm = np.zeros((70, 306, 3), dtype=np.float32)
# HM = np.zeros((70, 306, 2), dtype=np.float32)
# joint = []
# position = []
# for line in input_file:
#     line = line.strip()
#     line = line.split(' ')
#     joint.append(line[-4])
#     position.append(line[1:-4])
#
# print(joint)
# print(position)
# for i in range(len(joint)):
#     if int(joint[i]) == 1:
#         num_positions = len(position[i])
#         temp = hm[:, :, int(joint[i])]
#         img = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
#         for j in range(num_positions//2 - 1):
#             print(num_positions//2 - 1)
#             print("cishu")
#             cv2.line(img, (int(position[i][2 * j])//8, int(position[i][2 * j + 1])//8), (int(position[i][2 * j + 2])//8,
#                                                                                          int(position[i][2 * j + 3])//8), (255, 255, 255), 1)
#         hm[:, :, int(joint[i])] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# hm = cv2.GaussianBlur(hm, (3, 3), 0)
#
# def _makeGaussian(height, width, sigma=3, center=None):
#     """ Make a square gaussian kernel.
#     size is the length of a side of the square
#     sigma is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """
#     x = np.arange(0, width, 1, float)
#     y = np.arange(0, height, 1, float)[:, np.newaxis]
#     if center is None:
#         x0 = width // 2
#         y0 = height // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]
#     return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)
#
#
# for i in range(len(joint)):
#     if int(joint[i]) == 0:
#         num_positions = len(position[i])
#         temp = hm[:, :, int(joint[i])]
#         img = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
#         for j in range(num_positions//2):
#             print(num_positions//2)
#             print("cishu")
#             print(int(position[i][2*j])//8)
#             print(int(position[i][2*j+1])//8)
#             hm[:, :, int(joint[i])] += _makeGaussian(70, 306, sigma=7, center=(int(position[i][2*j])//8, int(position[i][2*j+1])//8))
#
# HM[:, :, 0] = hm[:, :, 0] + hm[:, :, 1]
#
#
#
# cv2.namedWindow('hm0')
# cv2.imshow('hm0', hm[:, :, 0])
# cv2.namedWindow('hm1')
# cv2.imshow('hm1', hm[:, :, 1])
# cv2.namedWindow('gtmap')
# cv2.imshow('gtmap', hm[:, :, 0]+hm[:, :, 1])
# cv2.namedWindow('HM')
# cv2.imshow('HM', HM[:, :, 0])
# cv2.waitKey(0)

# hm[1, 1, 0] = 255
# hm[1, 1, 1] = 255
# hm[1, 1, 2] = 255
##################      NMS     ###############
# sup_row = 3
# sup_col = 3
# hm_pad = np.zeros((90+sup_row-1, 160+sup_col-1, 4), dtype=np.float32)
# for i in range(hm.shape[2]):
#     for row in range(hm.shape[0]):
#         for col in range(hm.shape[1]):



import  time
start = time.time()
pic_folder = "."
pic = cv2.imread(pic_folder + '/' + '224x172_depth_20180903_160449' + '.jpg')
pic = cv2.imread('./224x172_depth_20180903_160449.jpg')
# pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
# pic = cv2.GaussianBlur(pic, (15, 15), 0)
print(pic.shape)
cv2.namedWindow('picture')
cv2.imshow('picture', pic)
print(time.time()-start)
cv2.waitKey(1000)

file_read = open(pic_folder + '/' + '224x172_depth_20180903_160449' + '.raw', 'rb')


for i in range(224*172):
    content=file_read.read(4)
    print( content)
    arr=bytearray(content)
    number=float(arr[3]<<24+arr[2]<<16+arr[1]<<8+arr[0])
    # a=float.fromhex(content.hex)
    print( number)
file_read.close()


# hm[:, :, 0] = cv2.fromarray(hm[:, :, 0])
