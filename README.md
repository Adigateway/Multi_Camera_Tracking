# Multi-Camera Person Tracking with Top-Down Localization

This project addresses the problem of tracking people from multiple synchronized CCTV cameras and projecting their movement into a unified top-down view of the room. The system ensures consistent global identity tracking across all views and visualizes their position in real time.

## Features

- Supports multiple CCTV camera inputs
- Real-time person detection using YOLOv8
- Homography-based projection to a 2D top-down room map
- Consistent global ID assignment using a custom tracker
- Clustering to eliminate duplicate detections across cameras
- Ready for extension with posture classification (sitting vs standing)

## How It Works

1. YOLOv8 detects people in each camera feed
2. Each detection is mapped into the top-down room using camera-specific homography matrices
3. Clustering is applied to remove duplicate detections of the same person from multiple cameras
4. A custom global tracker assigns unique IDs and maintains trajectories
5. The system displays all tracked people on a single, clean top-down layout

## Folder Structure
project/
├── yolo_trackdown_1.py         # Main tracking and visualization script
├── train.py                    # Training script for posture classification (planned)
├── tracking/
│   ├── custom_tracker.py       # Global identity tracker
│   └── utils.py                # Utility functions for tracking
├── homographies/               # Numpy files for each camera’s homography matrix
├── videos/                     # Input video files (not included in this repo)
