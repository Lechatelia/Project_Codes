import numpy as np
import cv2
import Dynamic

htmap_root = '/home/lechatelia/Desktop/CPM-boundary-data/145538AA/stage6_htmap/'

test_img = '/home/lechatelia/Desktop/CPM-boundary-data/145538AA/stage6_htmap/pred_lane-238.jpg'

img = cv2.imread(test_img)
img_show = Dynamic.Generate_Route(img)


cv2.namedWindow('img')
cv2.imshow('img', img)
cv2.namedWindow('img_show')
cv2.imshow('img_show', img_show)
cv2.waitKey(0)