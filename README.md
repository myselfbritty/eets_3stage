
S3NET: Instance Level Segmentation
===========================================

Overview
--------
Computer-assisted medical surgeries require accurate instance segmentation of surgical instruments in the endoscopic camera view to carry out many downstream perception and control tasks. There are no instance segmentation dataset available for instruments in Neuroendoscopy. We propose a new endoscopic endonasal transsphenoidal surgery (EETS) dataset for pituitary adenomas. We observe that, due to the typical orientation and aspect ratio of surgical instruments, the cross-domain fine-tuning of the instance segmentation model detects and segments the object regions correctly but is insufficient to classify the segmented regions accurately. 
We propose a novel three-stage deep neural network framework to augment a third stage in a standard instance segmentation pipeline to perform mask-based classification of the segmented object. To handle small datasets with visually similar classes, we train the proposed third stage using ideas from metric learning.
Read more: [Paper]()

Data
----
EETS consisting of 20 X 125-frame sequences is used as train set, 10 X 125-frame sequences as val set and 10 X 125-frame sequences is used as test set.
Instrument labels are 
suction
irrigation
spachula
scissors
knife
navigation
biopsy1
curette
drill
tumor_biopsy


Method
------
S3NET
The basic S3NET pipeline is divided into two:
1) Stage_1_2
3) Stage 3

These two are set on different environments:

Installation: Stage_1_2
------------
To install you can run
(Tested on Ubuntu 16.04. For Ubuntu 18 and 20, install gcc 9)

* conda create -n S3NET_Stage_1_2 python=3.7
* conda activate S3NET_Stage_1_2
* conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.2 -c pytorch
* conda install cython
* pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
* pip install mmcv==0.2.14
* pip install tqdm
* pip install opencv-python
* pip install scikit-image
* git clone https://github.com/S3NET/A_Three_Stage_Deep_Neural_Network_for_Highly_Accuracte_Surgical_Instrument_Segmentation.git S3NET
* cd S3NET
* bash compile.sh
* python setup.py install
* pip install .

Installation: Stage 3
------------
To install you can run
(Tested on Ubuntu 16.04. For Ubuntu 18 and 20, install gcc 9)
Dependecies:
Nvidia Driver >= 460
CUDA == 11.2
Cudnn == 8.1

* conda create -n S3NET_Stage_3 python=3.7
* conda activate S3NET_Stage_3
* conda install cython
* pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
* pip install tensorflow==2.5
* pip install mmcv==0.2.14
* pip install tqdm
* pip install opencv-python
* pip install sklearn
* pip install matplotlib


Organize data
----------

Here we describe the steps for using the Endoscopic Vision 2017 [1] for instrument-type segmentation.

Download the EETS dataset from [Link](https://drive.google.com/drive/folders/18_w_-aLx55XHfqP1fAlbKNksAaJ45_xm?usp=sharing>). Arrange the data in the folder format

::

    ├── data
    │   ├── EETS
    │       │── train
    │           ├── annotations
    │           ├── binary_annotations
    │           ├── images
    │           ├── segms
	│
    │   	│── test
    │           ├── annotations
    │           ├── binary_annotations
    │           ├── images
    │           ├── segms
	│
    │       .......................

		
Stage 1_2
------------------------------

- Organize the data of the dataset into the appropriate splits for coco format.

Copy dataset at data/EETS/train

Training
---------------
Resized weights are in pre-trained-weights folder

``python training_routine.py``

Testing
----------
Run the testing by 

``python testing_routine.py``

At this point, Stage 1 and 2 is over and we need to now improve the classification of the instances generated.

Stage 3
----------

Training
---------------
Resized weights are in pre-trained-weights folder

``python train_mask_classifier.py``

Testing
----------
Run the testing by 

``python test_mask.py``


Evaluation
----------

Evaluation was performed using ISINet evaluation framework as mentioned in [1].


References

[1] González, Cristina, Laura Bravo-Sánchez, and Pablo Arbelaez. "Isinet: an instance-based approach for surgical instrument segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
