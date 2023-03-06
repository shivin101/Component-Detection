from asyncore import read
import os
import cv2 as cv
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import os
import sys


from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import *

#Run the network on the image arrays
#To do make this a function for custom images and parameters

config = read_config()
bdcn_path = config['FileConfig']['bcdn_path']
sys.path.append(bdcn_path)
import bdcn

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(image,model):

    """
    Function to run a model on an image by converting it to a torch tensor
    Input: 
        image[Image Array], model[pytorch model]
    Output: 
        fuse[output of model on Image array image]
    """
    mode_cuda=1
    model.eval()
    if mode_cuda==1:
        model.cuda()
  
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    if mode_cuda==1:
        image = image.cuda()

    image = Variable(image, volatile=True)
    out = model(image)
    fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]

    return fuse


def run_network(image_arr,model):
    """
     Function to divide an image into fout segments and run a model on the 
    resultant divided tensor
    Input: image_arr[Image array usually in the form of a numpy array],
            model[pytorch model]
    Output: res[Fused image array after combinaiton] 
    """

    #Mean intensity for the board
    val = np.array([np.mean(sum(255*image_arr)[:,:,i]) for i in range(3)])
    res=[]
    for i in range(len(image_arr)):
        dst = np.uint8(255*image_arr[i])

        #Function to divide the image array into 4 quadrants for processing
        arr = divide_image(dst)
        part_res=[]
        for dst in arr:
            im = np.array(dst,dtype=np.float32)
            #Subtract the mean
            im -= val
            image = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
            part_res.append(test(image,model))

        #Function to join the 4 quadrants after processing
        res.append(join_image(part_res))
    return res


            
def get_edges_network(dst,model_path='../final-model/bdcn_pretrained_on_bsds500.pth'):
    
    """
    Function to load a model and its correponding weight and set it for processing.
    Input: dst[image array], mode_path[path to model weights]
    Output: res[resultant image array from running the network], 
            opening[resultant array after morphological operations]
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    #Load model and saved weights
    model = bdcn.BDCN()
    model.load_state_dict(torch.load(model_path))
    
    
    #Run the network
    res = run_network(dst,model)
    combined_edge= np.uint8(res[0]+res[1]+res[2])
    combined_edge[combined_edge>0]=1
    
    #Apply morphological operations
    opening = np.uint8(255*combined_edge)
    opening = apply_morphing(opening)
    
    
    return res,opening

def get_solders(img_arr):
    req = np.uint8(deepcopy(img_arr[0])*255)
    req[req<170]=0
    cv.imshow('solders',req)
    cv.imshow('original',img_arr[2])
    cv.waitKey(0)
    cv.destroyAllWindows()
    