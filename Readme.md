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


The main algorithm is defined in `main.py` 

MS-COCO based evalutaions and other data and plotting utilities are included in the `utils` folder

The configuration for running the network is defined inside `configurations.ini` 

`network.py` defines how to train and evaluate a network and `tf_record.py` defines operations related to tensorflow operations on MS-COCO based datasets

 The rest of the files are self descriptive to a large extent

### Results

#### Edge Detection
![Edge detection using BCDN](boards/4.png?raw=True)
![Edge detection results](boards/3.png?raw=True)


#### Component Detection Results
![Component Board1](boards/1.png?raw=True)
![Component Board2](boards/2.png?raw=True)

