# Exam-Cheating-Detection
# AI-Driven Exam Cheating Detection System (Prototype)

## Problem Statement
Cheating during online exams undermines fairness and authenticity. This system aims to detect cheating attempts in real-time by analyzing video feed for suspicious behavior using AI.

## Key Features
- Real-time person detection using YOLOv5.
- Simulated suspicious behavior detection.
- Sequence analysis (LSTM simulation) to flag cheating.
- Visual feedback with bounding boxes and color-coded status.
- Simple and extensible prototype for further improvements.

## Technologies Used
- Python 3.x
- PyTorch (for YOLOv5 model)
- OpenCV (for video capture and visualization)
- NumPy

## How to Run
1. Clone the repo.
2. Install dependencies:
3. Run the main script:
4. Webcam window will open. Press 'q' to quit.

## Future Improvements
- Integrate real head pose/gaze detection.
- Train LSTM on actual behavior sequences.
- Add audio-based cheating detection.
- Build web dashboard for monitoring.
