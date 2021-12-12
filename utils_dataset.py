import os
import json
import glob
import shutil
import numpy as np
import cv2 as cv
import requests
import random
import time

# makes dataset 
def make_dataset(classes, json_file, src_dir, dst_dir):

    if not os.path.exists(dst_dir): os.mkdir(dst_dir)
    with open(str(json_file)) as annotation:
        # Load items in json
        data = json.load(annotation)
        annots = data["annotations"]
        for annot in annots:
            id = annot["image_id"]
            
            cat = annot['category_id']


            if cat in classes:
               
                
                jpg_file = os.path.join(src_dir, f'{id}.jpg')
                
                try:
                    shutil.copy(jpg_file, dst_dir)
                except:
                    print(f'could not copy {id}')            

# classes = [5,16,20,21,22]
# json_file = 'darknet/custom_cfg/wildlife.json'


# src_dir = 'darknet/custom_dataset'
# dst_dir = 'darknet/small_dataset'
# make_dataset(classes, json_file, src_dir, dst_dir)
# 

# copies labels.txt files over to img_dir
def copy_label(img_dir, label_dir):

    img_files = os.listdir(img_dir)
    for img_file in img_files:
        id = img_file.split('.')[0]
        label_file = label_dir + f'{id}.txt'
        try:
            shutil.copy(label_file, img_dir)
        except:
            print(f'could not copy {id}.txt')   

# img_dir = './darknet/small_dataset/'
# label_dir = './darknet/custom_dataset/'
# copy_label(img_dir, label_dir)



def change_label(src_dir, dict):
    files = os.listdir(src_dir)
    for f in files:
        if f.endswith('.txt'):
            ff = np.loadtxt(os.path.join(src_dir, f))
            label = ff[0]
            if label in dict.keys():

                ff = ff[1:]
                # print('ff', ff)
                # f = open(f, "w")

                f = open(os.path.join(src_dir, f), "w")
                # print('writing file', id)
                f.write(f'{dict[label]} ')
                f.write(' '.join([str(i) for i in ff]))
                f.close()
                # exit(-1)

# src_dir = './darknet/small_dataset'
# change_label(src_dir,{5:0,16:1,20:2,21:3,22:4})





def create_test_imgs(classes, json_file, src_dir, dst_dir):

    if not os.path.exists(dst_dir): os.mkdir(dst_dir)
    with open(str(json_file)) as annotation:
        # Load items in json
        data = json.load(annotation)
        annots = data["annotations"]
        for annot in annots:
            id = annot["image_id"]
            cat = annot['category_id']


            if cat in classes:
                
                jpg_file = os.path.join(src_dir, f'{id}.jpg')
                try:
                    shutil.copy(jpg_file, dst_dir)
                except:
                    print(f'could not copy {id}')            

# classes = [0,8,15,16,19,22]

# print(classes)
# json_file = 'darknet/custom_cfg/wildlife.json'


# src_dir = 'darknet/custom_dataset'
# dst_dir = 'test_dir/'
# create_test_imgs(classes, json_file, src_dir, dst_dir)


def write_train_test(data_dir, cfg_file):
    data_dir= 'small_dataset'
    cfg_dir = 'small_cfg'
    # Percentage of images to be used for the test set
    percentage_test = 20
    # Create and/or truncate train.txt and test.txt
    file_train = open(cfg_dir+'/train.txt', 'w')  
    file_test = open(cfg_dir+'/test.txt', 'w')
    print('train.txt', file_train)
    # Populate train.txt and test.txt
    counter = 1  
    index_test = round(100 / percentage_test)  
    for pathAndFilename in glob.iglob(os.path.join(data_dir, "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
            counter = 1
            file_test.write(data_dir+ "/" + title + '.jpg' + "\n")
        else:
            file_train.write(data_dir+ "/" + title + '.jpg' + "\n")
            counter = counter + 1

# data_dir = 'small_dataset'
# cfg_dir = 'small_cfg'
# write_train_test(data_dir, cfg_dir)




# def copy_small_subset():
#     root_dir = './darknet'
#     test_data = os.path.join(root_dir, 'small_cfg/test.txt')
#     imgs = []
#     with open(test_data) as file:
#         for f in file:
            
#             imgs.append(os.path.join(root_dir, f.strip()))

#     imgs = reversed(sorted(imgs))
    
    
#     for im in imgs:
#         im_num = im[1:].split('.')[0][-4:]
#         print(im_num)
#         frame = cv.imread(im)

#         # cv.imshow('Window', frame)

#         # key = cv.waitKey(1000)#pauses for 3 seconds before fetching next image
#         # if key == 27:#if ESC is pressed, exit loop
#         #     cv.destroyAllWindows()
#         #     break

# copy_small_subset()

