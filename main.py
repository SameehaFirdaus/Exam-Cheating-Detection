import cv2
import torch
import numpy as np
from collections import deque

# Load YOLOv5 pretrained model (from ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define suspicious behaviors as detected classes or keypoints (simulate)
SUSPICIOUS_CLASSES = ['person']  # We'll detect 'person' and track head direction in demo

# Simulate sequence memory for LSTM input (here, just last N frames suspicious flags)
SEQUENCE_LENGTH = 10
suspicious_sequence = deque(maxlen=SEQUENCE_LENGTH)

# Fake LSTM decision threshold for demo (number of suspicious frames in sequence)
CHEATING_THRESHOLD = 5

def is_suspicious_behavior(detections):
    """
    Simulate suspicious behavior detection from YOLO detections.
    For prototype, if person detected looking away from camera (simulate),
    flag suspicious.
    """
    # In real case, you'd use head pose estimation or keypoint analysis
    # Here, we randomly flag suspicious behavior for demo
    if len(detections) > 0:
        # Simulate 40% chance suspicious per frame
        return np.random.rand() < 0.4
    return False

def lstm_decision(seq):
    """
    Simulated LSTM: if suspicious behavior count in last N frames > threshold, cheating detected.
    """
    count = sum(seq)
    return count >= CHEATING_THRESHOLD

def main():
    cap = cv2.VideoCapture(0)  # webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO object detection
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # bounding boxes with scores
        
        # Filter detections by class 'person' (class 0 in COCO)
        person_detections = [d for d in detections if int(d[5]) == 0 and d[4] > 0.5]
        
        # Check suspicious behavior in this frame (simulate)
        suspicious = is_suspicious_behavior(person_detections)
        suspicious_sequence.append(int(suspicious))
        
        # LSTM decision
        cheating_detected = lstm_decision(suspicious_sequence)
        
        # Draw bounding boxes and labels
        for det in person_detections:
            x1, y1, x2, y2, conf, cls = det
            color = (0,255,0) if not cheating_detected else (0,0,255)  # green or red
            label = "Not Cheating" if not cheating_detected else "Cheating"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show info
        info_text = f"Cheating: {'YES' if cheating_detected else 'NO'}"
        info_color = (0,0,255) if cheating_detected else (0,255,0)
        cv2.putText(frame, info_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, info_color, 3)
        
        cv2.imshow('AI-Driven Exam Cheating Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
