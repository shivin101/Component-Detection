import os
import cv2 as cv
import numpy as np
from copy import deepcopy


def find_canny(img_arr):
    """
        Function to denoise and get canny edges for an array
        Input:: N dimensional image array:img_arr
        Outpt:: N dimensional binary edge masks for each image: edges
    """	
    edges=[]
    for ver in range(len(img_arr)):
        image_arr = img_arr[ver] 
        edges.append([])
        for i in range(len(image_arr)):

            img = np.uint8(255*image_arr[i])
            img = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

            #Bilateral filtering
            img = cv.bilateralFilter(img,20,100,100)
            img = cv.bilateralFilter(img,20,100,100)

            #Gaussian Blurring the image
            img = cv.GaussianBlur(img,(5,5),0)
            img = cv.GaussianBlur(img,(5,5),0)
            edge_canny = cv.Canny(img,10,255)
            
            
            edges[ver].append(edge_canny)
    return edges



def canny_detector(edges):
    """
        Function to denoise and get canny edges for an array
        Input:: N dimensional image array:edges
        Outpt:: N dimensional binary edge masks for each image: canny_edge_arr

    """
	
    canny_edge_arr=[]
    for ver in range(len(edges)):
        edge_map = deepcopy(edges[ver])
        edge_sum_1 = edge_map[1]+edge_map[2]
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        kernel3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        opening = deepcopy(edge_sum_1)
        opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel2)
        opening = cv.morphologyEx(opening, cv.MORPH_TOPHAT, kernel3)
        
        opening[opening>0]=1
        canny_edge_arr.append(opening)
    return canny_edge_arr
    