import os
import cv2 as cv
import numpy as np
from copy import deepcopy
from utils.utils import *
from dsu import *

def thresh_data(height_map):
    "Function to call find_thresh given a height height_map"
    "Input:" 
    "height_map: ND numpy array "
    "Output: ret2:NDimensional numpy array,dat_copy_low: NDimensional numpy, data_copy_high(numpy array)"
    " array containing onjects at high threshold and low threshold"
    ret2,th2 = cv.threshold(height_map,height_map.min(),height_map.max(),cv.THRESH_BINARY+cv.THRESH_OTSU)
    upper_thresh_low = ret2+120
    upper_thresh_high = ret2-30
    _,data_copy_low=find_thresh(height_map,ret2,upper_thresh_low)
    _,data_copy_high=find_thresh(height_map,ret2,upper_thresh_high)
    return ret2,data_copy_low,data_copy_high


def find_thresh(height_map,lower_lim,upper_thresh):
    """
    Function that threshold a height data map based on a seed upper and lower threshold
    Thresholding is based on area percentage and not threshold defined
    Input:: NDimensional NUmpy array height_map, Int: lower_lim, Int: Upper_thresh
    Output:: Int: Threshold value, Thresholded height map
    """
    print((height_map.size))
    pix_count = height_map.size
    thresh = 20
    perc=0.15
    flag=1
    iter_val = 1
    while flag:
        
        lower_thresh = lower_lim-thresh
        #Create a copy of data
        data_copy = deepcopy(height_map)
        data_copy[data_copy>upper_thresh]=0
        data_copy[(data_copy<=upper_thresh) & (data_copy>lower_thresh)]=255
        data_copy[data_copy<=lower_thresh]=0
        #Find area currently non zero
        covered = (sum(sum(data_copy>0))*1.0/pix_count)

        #Increase the threshold based on if the area is coverage 
        #is outside bounds
        if  covered>perc:
            flag=0
            thresh-=iter_val
            lower_thresh = lower_lim-thresh
            data_copy = deepcopy(height_map)
            data_copy[data_copy>upper_thresh]=0
            data_copy[(data_copy<=upper_thresh) & (data_copy>lower_thresh)]=255
            data_copy[data_copy<=lower_thresh]=0
        else:
            thresh+=iter_val
    thresh_val = lower_lim-lower_thresh
    return thresh_val,data_copy
           
def iterative_thresh(height_map,start_thresh=70,iter_val=100)->list:
    """
    Function to iteratively threshold the data and circle out 
    components based on their appearance at different levels of thresholds
    Input:: Ndimensional Array: height_map, Int: start_thresh, Int:iter_val
    Output:: rect_list(List): List of rectangular bounding boxes for the components at various thresholds
    """
    combined_rect=[]
    #Threshold upto 70% of the max value in height map
    
    for thresh in np.arange(start_thresh,0.7*height_map.max(),iter_val):
        print(thresh)
        upper_thresh = thresh+iter_val
        lower_thresh = thresh
        data_copy = deepcopy(height_map)
        data_copy[data_copy>upper_thresh]=0
        data_copy[data_copy<=lower_thresh]=0
        data_copy[data_copy!=0]=2000

        _,_,rect_list = get_bounding_box(data_copy,area_upper_lim=120000,area_lower_lim=100)
        combined_rect = combined_rect+rect_list
        print(len(rect_list))
        
    return combined_rect

def refine_bb(rect_list,coverage_lim = 3,overlap_lim =0.4)->list: 
    """
        Function to refine a list of bounding boxes using DSU algorithm
        Input: 
               rect_list(List[data]): list of rectangular bounding boxes.
               converage_lim(Int): coverage

        Output: 
            refined_list(List[bounding boxes]) list of rectangular bounding boxes.
    """
    

    refined_list = []


    #Sort According to the area of the bounding box


    rect_dsu = DSU(len(rect_list))
    for i in range(len(rect_list)):
        
        rect = rect_list[i]

        for j in range(i+1,len(rect_list)):
            ref_rect = rect_list[j]

            #calculate overlap for rectangles
            overlap,coverage= Rectangular_intersection(rect,ref_rect)
 

            if overlap>overlap_lim and overlap<=1.0 and coverage>=coverage_lim:
                rect_dsu.union(i,j)
                
                print("Union:",coverage,overlap)

            elif coverage<coverage_lim and overlap >overlap_lim and rect_area(ref_rect)>3000:
            #If the coverage exceed some ammount deconstruct the entire set 
            #constructed up until this point

                print("Dissolve:",coverage,overlap)
                for k in range(len(rect_list)):
                    if rect_dsu.parent(k)==j:
                        rect_dsu.set_parent(k,k)

            else:
                pass



    for i in range(len(rect_list)):
        if rect_dsu.parent(i)==i:
            refined_list.append(rect_list[i])


    return refined_list    