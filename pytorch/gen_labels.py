import random
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

def get_labels(mode):
#mode = 'train'
    
    if mode == 'train':
        with open('/misc/lmbraid19/mittal/dense_prediction/forked/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') as f:
            contents = f.readlines()
            contents = [x.strip() for x in contents]
            #contents = contents[:732]
    else:
        with open('/misc/lmbraid19/mittal/dense_prediction/forked/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt') as f:
            contents = f.readlines()
            contents = [x.strip() for x in contents]
        

    class_list = ['aeroplane',
                    'bicycle',
                    'bird',
                    'boat',
                    'bottle',
                    'bus',
                    'car',
                    'cat',
                    'chair',
                    'cow',
                    'diningtable',
                    'dog',
                    'horse',
                    'motorbike',
                    'person',
                    'pottedplant',
                    'sheep',
                    'sofa',
                    'train',
                    'tvmonitor']

    if mode == 'train':
        out_img = open('/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/train_img.txt', 'w')
        out_label = open('/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/train_label.txt', 'w')
    else:
        out_img = open('/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/val_img.txt', 'w')
        out_label = open('/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/val_label.txt', 'w')

    labels = np.zeros((len(contents), 20))
    #labels[:,20] = 1.0

    for i, item in enumerate(contents):
        xml_contents = []
        xml_file = '/misc/lmbraid19/mittal/dense_prediction/forked/models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/Annotations/' + item + '.xml'
           
        tree = ET.parse(xml_file) 
        root = tree.getroot()
        ranks = [] 
        #indexes = []
        for country in root.findall('object'):
            rank = country.find('name').text
            ranks.append(rank)
        classes = list(set(ranks))
            
        out_img.write("%s.jpg\n" % item )
        
        for class_found in classes:
            index = class_list.index(class_found) 
            #indexes.append(index)
            #out_label.write("%i " % (index) )
            labels[i][int(index)] = 1
        for lab in range(20):
            out_label.write('%i ' % (labels[i][lab]))
        out_label.write("\n")        
    #print (labels)
    print (np.sum(labels, axis=0))
    return labels
        
