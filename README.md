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

```text
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
