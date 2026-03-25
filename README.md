# Multimodal Emotion Recognition for Safety Applications

[cite_start]This project implements a real-time multimodal emotion recognition system that integrates facial expression analysis and hand gesture recognition to detect distress and trigger emergency alerts[cite: 16, 110].

## Overview
[cite_start]The system uses a webcam to capture live video[cite: 92, 121]. It processes frames to:
1.  [cite_start]**Detect Faces:** Using a Haar Cascade classifier[cite: 93, 152].
2.  [cite_start]**Recognize Emotions:** Using a pre-trained ONNX CNN model (FERPlus) to identify distress indicators like 'Sad' and 'Fear'[cite: 17, 185, 189].
3.  [cite_start]**Analyze Hand Gestures:** Using the Convex Hull algorithm and contour detection to identify distress-related gestures like a clenched fist[cite: 18, 96, 113].
4.  [cite_start]**Multimodal Fusion:** An alert is only triggered when both a distress emotion and a specific hand gesture are detected simultaneously, reducing false positives[cite: 19, 48, 128].

## System Architecture
[cite_start]The system operates as a modular real-time processing pipeline[cite: 120, 145]. 

## Setup and Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/bufferizer25/Multimodal-Emotion-Recognition-for-safety-applications.git](https://github.com/bufferizer25/Multimodal-Emotion-Recognition-for-safety-applications.git)
    cd facial-emotion-detection
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the Model:**
    [cite_start]Ensure `emotion-ferplus-8.onnx` is in the root directory.

## Usage
- [cite_start]**Calibration:** Run `python calibrate.py` to adjust the HSV skin color range for hand detection in your lighting environment.
- [cite_start]**Main System:** Run `python Main_Code.py` to start the real-time detection.
- [cite_start]**Termination:** Press 'Q' to exit the application[cite: 179].

## Citation
If you use this work, please cite our IEEE paper:
> [cite_start]Thota, A., Sri, N. D., Prasannatha, A., & Dinesh, K. "Multimodal Emotion Recognition Using Facial Expression and Hand Gesture Analysis for Safety Applications"[cite: 1, 3, 7, 11, 33].
