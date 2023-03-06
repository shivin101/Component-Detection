import os
import cv2 as cv
import numpy as np
from copy import deepcopy
import configparser




# Method to read config file settings
def read_config():
    config = configparser.ConfigParser()
    config.read('../config/configurations.ini')
    return config

def apply_morphing(image_arr,kernelSize1=5,kernelSize2=3):
    """
    #Apply morphological operations to 
    #Make the task of segmentation easier
    Input: 
    """
    #Declare Structuring elements
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel1,kernel1))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kernel2,kernel2))


    #Apply morphology operations
    opening = deepcopy(image_arr)
    opening = cv.morphologyEx(opening, cv.MORPH_ERODE, kernel2)
    opening = cv.morphologyEx(opening, cv.MORPH_ERODE, kernel)
    return opening

def get_bounding_box(edge_map,color_map=(0,255,0),area_upper_lim=90000,area_lower_lim=100,height_lim=5,width_lim=5):
    
    """
    Function to get a list of bouding boxes given the edge of height map
    
    Input:: edge_map[N dimemsional numpy array],color_map[BGR color map],
     area_upper_lim[int],area_lower_lim=[int],height_lim=[int],width_lim=[int] 

    Output:: contours[list of extracted contours],thresh_img[],
             rect_list[list of bouding box rectangles]
    """
    _, threshed_img = cv.threshold(np.uint8(edge_map),
                    0, 255, cv.THRESH_BINARY)
    
    thresh_img = deepcopy(threshed_img)
    contours, _= cv.findContours(threshed_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)

    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    keypoints = []
    rect_list = []
    area_lim = 2000
    w_h_ratio = 0.1
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv.boundingRect(c)
        # draw a colored rectangle to visualize the bounding rect
#         cv.rectangle(img, (x, y), (x+w, y+h), color_map, 2)

        # get the min area rect
        rect = cv.minAreaRect(c)
        area_rect = w*h
        if area_rect<area_upper_lim and area_rect>area_lower_lim and rect[1][0]>height_lim and rect[1][1]>width_lim:
            
            #Criterial for eliminating outlier bounding boxes
            if (area_rect<area_lim and h/w>w_h_ratio and w/h>w_h_ratio) or (area_rect>=area_lim):
                box = cv.boxPoints(rect)

                
                # convert all coordinates floating point values to int
                box = np.int0(box)
                
                rect_list.append([x,y,w,h])
               
                # finally, get the min enclosing circle
                (x, y), radius = cv.minEnclosingCircle(c)
                cv.KeyPoint(x,y, _size=radius,_angle=rect[2] )
                
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)

     

    
    return contours,thresh_img,rect_list

def divide_image(image):
    """
    Utility function to run edge detection network for large images
    """
    h,w,_=image.shape
    image_arr=[]
    print(image.shape) 
    im1 = image[0:h//2,0:w//2,:]
    im2 = image[0:h//2,w//2:w,:]
    im3 = image[h//2:h,0:w//2,:]
    im4 = image[h//2:h,w//2:w,:]
    
    image_arr=[im1,im2,im3,im4]
    return image_arr

def join_image(image_arr):
    """
    Utility function to run the edge detection network for large images
    """
    im1=np.hstack((image_arr[0],image_arr[1]))
    im2=np.hstack((image_arr[2],image_arr[3]))
    image = np.vstack((im1,im2))
    print(image.shape)
    return image

def sort_rect(rect_list):
    area = []
    for rect in rect_list:
        area.append(rect_area(rect))
    idx = np.argsort(np.array((area)))
    return idx

#Calculate the limits of intersection for two Lines
def calculateIntersection(a0, a1, b0, b1):
    
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0

    return intersection

#Calculate the limits of intersection for two rectangles
def Rectangular_intersection(rect1,rect2):
    [X0, Y0, X1, Y1] = rect1
    AREA = float((X1) * (Y1 ))
    [x0, y0, x1, y1]  = rect2       
    width = calculateIntersection(x0, x0+x1, X0, X0+X1)        
    height = calculateIntersection(y0, y0+y1, Y0, Y0+Y1)        
    area = width * height
    rect2_area = x1*y1
    percent = area / AREA
    if area!=0:
        percent2 = rect2_area/AREA
    else:
        percent2 = 0
    return percent,percent2
    
#Function to return area of a rectangle    
def rect_area(rect):
    [X0, Y0, X1, Y1] = rect
    area = float((X1) * (Y1 ))
    return area

def draw_rects(rect_1,rect_2):
    img = deepcopy(images[2])
    for [x,y,w,h] in [rect_1,rect_2]:
        cv.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    cv.imshow("container", img)
    while True:
        key = cv.waitKey(9)

        if key !=-1: #ESC key to break
            break

    cv.destroyAllWindows()
    
def draw_rect(rect,img):
    for [x,y,w,h] in rect:
        cv.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    cv.imshow("container", img)
    while True:
        key = cv.waitKey(9)

        if key !=-1: #ESC key to break
            break

    cv.destroyAllWindows()