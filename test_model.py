#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 11:39:15 2021

@author: dxtr
"""

import json
import os
#from model import make_classifier
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import cv2
from sys import exit
import tensorflow as tf
import random
tf.keras.backend.set_image_data_format('channels_first')

ip_path = './output/stage_2/'
ip_file = 'EETS_MaskRCNN_EETS_post_ep_12.pkl.segm.json'
op_path = './output/stage_3/'
annotation_file = './data/test/segms/EETS_test.json'
op_file = 'stage_3.json'
fold = './data/test'
fold_num = 0

image_path_dict = {}


# print('\n\n\n\n\n\n\n\n\n\n\n\#######################################################\n\n\n\nn\n\n\n\n\n\')



model = make_classifier(num_classes=10)
b = np.zeros((1,1,56,56))
ip2 = tf.constant(b)
a = np.zeros((1,3,224,224))
ip = tf.constant(a)
y = np.zeros((1,10))
label = tf.constant(y)
op = model([ip,ip2,label]).numpy()

model.load_weights('./pre-trained-weights/Stage_3/maskrcnn.h5')
# print('hoooray')
# exit()
for image in os.listdir(os.path.join(fold,'images')):
    current_image_path = os.path.join(fold,'images',image)
    
    image_path_dict[image] = current_image_path
    
example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
category_names = set([category['supercategory'] for category in categories])
category_ids = example_coco.getCatIds(catNms=['square'])


image_ids = example_coco.getImgIds(catIds=category_ids)

# print(image_ids)
image_id_dict = {}

for image_id in range(len(image_ids)):
    image_data = example_coco.loadImgs(image_ids[image_id])[0]
    # print(image_data)
    current_id = image_data['id']
    current_file = image_data['file_name']
    current_path = image_path_dict[current_file]
    image_id_dict[current_id]    = current_path
# print(image_id_dict)
# import sys
# sys.exit()

@tf.function
def predict_step(current_image, current_mask):
   op = model([current_image, current_mask], mode='softmax')
   return op

data_count = np.zeros((10,))
with open(os.path.join(ip_path, ip_file), 'r') as f:
    data = json.load(f)
data_write = []
orig_count_1 = 0
orig_count_2=0
new_count_1=0
new_count_2=0
for d in data:
    data_new = {}
    image_id = d["image_id"]
    # bbox = d["bbox"]
    score = d["score"]
    category_id = d["category_id"]
    seg = d["segmentation"]
    data_new["image_id"] = image_id
    # data_new["bbox"] = bbox 
    data_new["score"] = score
    data_new["category_id"] = category_id
    
    data_new["segmentation"] = seg
    gt = category_id

    annotation_ids = example_coco.getAnnIds(imgIds=image_id, catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)

    # current_data=np.zeros((224,224,4))
    current_mask = np.ones((1,1,56,56), dtype=np.float32)
    mask = maskUtils.decode(seg)
    mask = 255*np.ascontiguousarray(mask, dtype=np.uint8)
    mask = cv2.resize(mask, (56,56))
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    current_mask[0,0,:,:] = mask
    current_image = cv2.imread(image_id_dict[image_id])
    ht, wt = current_image.shape[:2]
    current_image = cv2.resize(current_image, (224,224))
    
    
    current_image = current_image.copy().astype(np.float32)
    mean=[123.675, 116.28, 103.53]
    mean = np.asarray(mean)
    std=[58.395, 57.12, 57.375]
    std = np.asanyarray(std)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(current_image, mean, current_image)
    cv2.multiply(current_image, stdinv, current_image)
    
    current_image = np.transpose(current_image, [2,0,1])
    current_image = np.expand_dims(current_image, axis = 0)
    op = predict_step(current_image, current_mask).numpy()
    new_id = np.argmax(op[0,:]) +1
    data_new["category_id"] = int(new_id)
    

resFile = os.path.join(op_path, op_file)
with open(resFile, 'w') as outfile:
    json.dump(data_write, outfile)


