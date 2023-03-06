# This file is used for extracting regions for training from image file given 
# the bounded boxes extracted in pre processing step


#Import Section
import os
import cv2 as cv
import numpy as np
from copy import deepcopy
from utils import *
from dsu import *
import random 
import pickle



config = read_config()
BOX_SIZE = int(config['PostProcess']['root'])
start = int(config['PostProcess']['root'])
NUM_REGIONS = int(config['PostProcess']['root'])
start = start*NUM_REGIONS



"""Function to extraction regions from an image regions given bounds"""
def extract_region(bounds,img,rects):
    count =start
    config = read_config()
    ROOT_DIR = config['PostProcess']['root']
    data_id = config['PostProcess']['data_id']
    while count<NUM_REGIONS+start:
        box = random_box(bounds)
        print(box)
        regions =random_box_extraction(img,box,rects)
        boxed_img = bounded_rects(img,box,regions,rects)
        
        save_file = ROOT_DIR+data_id+'_'+str(count)
        value_dict={}
        
        value_dict['image']=boxed_img[1]
        value_dict['boxes']=regions
        if value_dict and len(regions)>=1:
            count+=1
            with open(save_file, 'wb') as dict_items_save:
                pickle.dump(value_dict, dict_items_save)

def bounded_rects(img,box,rects,global_rects):
    boxed_img  = []
    
    for i in range(len(img)):
        temp_img=img[i][box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
        boxed_img.append(temp_img)

    rects = np.array(rects)
    temp_rects = deepcopy(rects)
    print(temp_rects)


    return boxed_img

"""Function to give a random box """
def random_box_extraction(img,bounding_box,rects,thresh=20):
    [x0, y0, x1, y1]  = bounding_box
    # print("Bounding Box is:",bounding_box)
    bounded_rects = []
    for rect in rects:
        [X0, Y0, X1, Y1] = rect
        start_x = (start_coord(x0,x1,X0,X1)-x0)
        end_x = (end_coord(x0,x1,X0,X1)-x0)
        start_y = (start_coord(y0,y1,Y0,Y1)-y0)
        end_y = (end_coord(y0,y1,Y0,Y1)-y0)
        area =(end_x-start_x)*(end_y-start_y)
        if area<=thresh or neg(start_x,start_y,end_x,end_y):
            continue
        else:

            bounded_rects.append(\
                [start_x,start_y,end_x,end_y])
    return bounded_rects

def neg(start_x,start_y,end_x,end_y):
    if((start_x<0) or (start_y<0)):
        return 1
    elif(end_x<0 or end_y<0):
        return 1
    else:
        return 0

def start_coord(bb_x,bb_len,x_0,x_w):
    if x_0<=bb_x:
        return bb_x
    elif x_0>=bb_x+bb_len:
        return -1
    else:
        return x_0
        
def end_coord(bb_x,bb_len,x_0,x_w):
    if bb_x+bb_len<=x_0+x_w:
        return bb_x+bb_len
    elif x_0+x_w<=bb_x:
        return -1
    else:
        return x_0+x_w


def random_box(bounds,size=BOX_SIZE):
    corner = []
    corner.append(random.randrange(bounds[0][0],bounds[0][1]))
    corner.append(random.randrange(bounds[1][0],bounds[1][1]))
    region  = [corner[0],corner[1],size,size]
    return region

def load_file(file_name):
    data_dict = np.load(file_name,allow_pickle=True)
    img = data_dict['image']
    rects = data_dict['boxes']
    return img,rects


if __name__ == "__main__":
    config = read_config()
    saved_root = config['PostProcess']['saved_root']
    data_id = config['PostProcess']['data_id']
    session_id = int(config['PostProcess']['session_id'])
    img,rects = load_file(saved_root+data_id+str(session_id)+'.txt')
    bounds = []
    bounds.append([BOX_SIZE,img[0].shape[0]-BOX_SIZE])
    bounds.append([BOX_SIZE,img[0].shape[1]-BOX_SIZE])
    extract_region(bounds,img,rects)