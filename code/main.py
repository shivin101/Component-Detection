import cv2 as cv
import numpy as np
import sys


from utils.utils import *
from canny import *
from thresh import *
from network import *
from dsu import *
from utils.histogram import *

import pickle



def main():
    # Load an example image
    #Perform config operations for setting up the file 
    config = read_config()
    root = config['FileConfig']['root']
    model_root = config['FileConfig']['model_root']
    array_index=2
    bdcn_path = config['FileConfig']['bcdn_path']
    sys.path.append(bdcn_path)
    image_dir =root+config['FileConfig']['img_dir']
    height_dir = root+config['FileConfig']['height_dir']
    img_arr=[]


    raws_list = sorted(os.listdir(height_dir))
    image_list = sorted(os.listdir(image_dir))
    count=0
    data_arr = []

    #Load data and images in an array
    for i in range(len(raws_list)):
        name=raws_list[i].split('.')[0]
        init_name = name[-7:]
        img_arr.append([])
    
        #Load the top mid and bottom image
        #and apply adaptive histogram equalization
        img = cv.imread(image_dir+init_name+'-Top.bmp')
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        img_arr[i].append(img_adapteq)
    
        img = cv.imread(image_dir+init_name+'-Mid.bmp')
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        img_arr[i].append(img_adapteq)
    
        img = cv.imread(image_dir+init_name+'-Bot.bmp')
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        img_arr[i].append(img_adapteq)
                
   
        #Load Height Data and reshape it to the size of the image
        with open(height_dir+raws_list[i],'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            print(len(data))
            array = np.reshape(data, img.shape[:2])
            data_arr.append(array)



    height_map = data_arr[array_index]
    img_map = img_arr[array_index]
    #Normalize Height data
    height_map = np.uint8(((height_map-height_map.min())/height_map.max())*255)
    ret,thr_low,thr_high = thresh_data(height_map)

    #Get edges from running neural network on the data
    res,combined_edge = get_edges_network(img_map,model_root)


    #get contours on the results fo the networks
    contours_edges,contour_canny,rect_list_edges = get_bounding_box(edge_map\
        =combined_edge,color_map=(0,0,255),area_upper_lim=50000,area_lower_lim=100)
    #Get bounding boxes from height data 
    rect_list_height = iterative_thresh(data_arr[array_index])

    #Get canny edges for finer details
    edges = find_canny(img_arr)
    canny_edge_arr = canny_detector(edges)

    #Refine edge detection results
    combined_list = refine_bb(rect_list_height)
    contours_edges,contour_canny,rect_list_edges = get_bounding_box(canny_edge_arr[array_index],\
        color_map=(0,0,255),area_upper_lim=20000,area_lower_lim=100)

    #Save the processed data
    save_dir =config['FileConfig']['save_dir']
    save_file = save_dir+root.split('/')[-1]+str(array_index)+'.txt'
    value_dict = {}
    value_dict['boxes']=combined_list
    value_dict['image']=img_arr[array_index]
    if value_dict:
        with open(save_file, 'wb') as dict_items_save:
            pickle.dump(value_dict, dict_items_save)


    # #Draw edges on the image data for visualization
    draw_rect(rect_list_edges,deepcopy(img_arr[array_index][1]))
    # #   print(len(rect_list_edges),len(rect_list_height))
    draw_rect(combined_list,deepcopy(img_arr[array_index][1]))


if __name__ == '__main__':
  main()