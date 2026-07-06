# Traffic-Sign-Detection-ORB-SIFT

Traffic sign detection project using Python and OpenCV.  
The project detects traffic signs from the GTSDB dataset using template matching with ORB and SIFT feature detectors.

## Project Objective

The objective of this project is to detect traffic signs in road images using classical computer vision techniques.

The project uses YOLO annotation files to crop traffic sign templates, then applies feature matching with ORB and SIFT to locate traffic signs inside scene images.

## Features

- Create traffic sign templates from YOLO label files
- Convert YOLO annotations into pixel bounding boxes
- ORB-based traffic sign detection
- SIFT-based traffic sign detection
- Feature matching using Lowe ratio test
- Homography estimation using RANSAC
- Detection box validation
- Run detection on single images
- Run detection on multiple images
- Save detection results as output images

## Technologies Used

- Python
- OpenCV
- NumPy
- ORB feature detector
- SIFT feature detector
- GTSDB dataset

## Project Structure


Traffic-Sign-Detection-ORB-SIFT/
│
├── create_Template.py
├── main.py
├── mainSift.py
├── orbDetectors.py
├── runOrbMultiple.py
├── RunSiftMultiple.py
├── SiftDetector.py
├── utils.py
├── templates/
├── results/
└── archive (1)/


Main Files: 
create_Template.py

This script creates traffic sign templates from the dataset.

It reads the YOLO label files, converts the normalized YOLO coordinates into pixel bounding boxes, crops the traffic signs from the images, and saves them inside class folders.

Output example:

templates/
│
├── class_0/
├── class_1/
├── class_2/
└── ...
orbDetectors.py

This file contains the ORB-based detection functions.

It performs:

ORB keypoint detection
ORB descriptor extraction
Feature matching using BFMatcher
Lowe ratio test
Homography estimation using RANSAC
Detection validation
Drawing the detected sign on the image
SiftDetector.py

This file contains the SIFT-based detection functions.

It performs:

SIFT keypoint detection
SIFT descriptor extraction
Feature matching
Lowe ratio test
Homography estimation
Detection validation
Drawing the detected sign on the image
runOrbMultiple.py

This script runs ORB detection on multiple images from the dataset.

Results are saved in:

results/detections_orb/
RunSiftMultiple.py

This script runs SIFT detection on multiple images from the dataset.

Results are saved in:

results/detections_sift/
utils.py

This file contains helper functions such as:

Converting YOLO labels to bounding boxes
Finding the matching label file for an image
Drawing bounding boxes on images
How It Works
The dataset images and YOLO labels are loaded.
YOLO labels are converted into pixel bounding boxes.
Traffic sign templates are cropped and saved by class.
ORB or SIFT features are extracted from both the template and the scene image.
Features are matched using a ratio test.
RANSAC is used to estimate a homography.
If the homography is valid, the traffic sign is detected.
The detected sign is drawn on the image.
The result is saved inside the results/ folder.
Installation

Install the required libraries:

pip install opencv-python opencv-contrib-python numpy
How to Run
1. Create templates
python create_Template.py
2. Run ORB detection
python runOrbMultiple.py
3. Run SIFT detection
python RunSiftMultiple.py
Dataset

This project uses the GTSDB traffic sign dataset.

Expected dataset structure:

archive (1)/
└── GTSDB_Train_and_Test/
    └── Train/
        ├── images/
        └── labels/
Output

The detection results are saved inside:

results/
│
├── detections_orb/
└── detections_sift/

Each output image contains the detected traffic sign with a bounding polygon and the number of matched features.

Notes
ORB is fast and works well for real-time feature matching.
SIFT is usually more robust but can be slower than ORB.
The quality of detection depends on the template quality, image resolution, and the number of valid feature matches.
Homography validation is used to reduce false detections.
Author

Abdelkarim Elmourad
