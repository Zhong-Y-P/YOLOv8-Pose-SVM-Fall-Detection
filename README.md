# YOLOv8-Pose-SVM-Fall-Detection
Real-time human fall detection based on YOLOv8-Pose and SVM, supports local camera and ROS2 deployment.

## Overview
This project implements a real-time human fall detection solution based on computer vision and machine learning.
We adopt YOLOv8n-Pose for human detection and pose keypoint extraction. Three types of features (aspect ratio, centroid velocity, normalized keypoints) are extracted and fed into an SVM classifier to distinguish fall behavior from normal activities.

The system supports local camera/video inference and ROS2 node deployment for robot system integration, which can be applied to elderly care and indoor safety monitoring.

## Tech Stack
- YOLOv8-Pose: Human pose estimation
- SVM (Scikit-learn): Binary classification
- OpenCV: Video capture and image processing
- Pandas & NumPy: Data processing
- ROS2: Robot operating system deployment

## Main Features
- Extract features from video datasets and generate CSV files for training
- Train, evaluate and save SVM classification model
- Real-time fall detection with local camera and visual display
- ROS2 publisher & subscriber nodes for status notification and alarm
- Model optimization by expanding datasets to reduce false detection
