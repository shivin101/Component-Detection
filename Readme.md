### Component Detection on PCB boards
## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* OpenCV
```
pip install python-opencv
```
* Scikit Learn and Scikit Image
```
pip install sklearn skimage
```
The algorithm for component detection can be summarised as 
1) Detect edges using BCDN network 
2) Use the edge map to identify components using contours as boundinb boxes
3) Use a trained Faster-RCNN to detect components on boards as bounding boxes
4) Refine the collection of bounding boxes from step 2 and 3 using a DSU

![Edge detection using BCDN](boards/4.png?raw=True)


The configuration for running the network is defined inside configurations.ini 
