# 🔍 YOLOv8 Pose-Based Fall Detection

A real-time human fall detection system using the **YOLOv8 Pose Estimation model** and OpenCV. This project uses body keypoints to detect whether a person has fallen, based on geometric heuristics from shoulder, hip, and ankle positions.

---


## 🧠 Fall Detection Logic

The fall detection logic is based on relative positions of keypoints:
- **Keypoints Used**: Left/Right Shoulder, Hip, and Ankle.
- **Confidence Threshold**: Only keypoints with confidence > 0.5 are considered.
- **Bounding Box Ratio**: If the bounding box is wider than it is tall, it's likely a horizontal (fallen) posture.
- **Vertical Heuristic**:
  - Shoulder is very close to the ankle (low vertical height).
  - Hip is below mid-point between shoulder and ankle.
  - Shoulder is close to or below the hip.
  - Combined with a wide bounding box, these indicate a fallen pose.




The fall is detected if any of the following conditions are true:

1. Shoulder Y > Ankle Y − shoulder_to_hip_length

2. Hip Y > Ankle Y − (0.5 × shoulder_to_hip_length)

3. Shoulder Y > Hip Y − (0.5 × shoulder_to_hip_length)

4. Bounding box height < width (horizontal person shape)
---




---

## 🚀 How It Works

1. **Load the YOLOv8 Pose Model**:
   Uses Ultralytics `yolov8n-pose.pt` for real-time human pose detection.

2. **Extract Keypoints**:
   Each detected person has 17 keypoints (x, y, confidence).

3. **Apply Fall Detection Heuristics**:
   Using distance and alignment between keypoints, we decide if the person has fallen.

4. **Visual Output**:
   - 🔴 Red keypoints plotted.
   - 🟥 Bounding box around each person.
   - 🔔 Alert message: `"Person Fell Down"` when detected.

---


## 🛠️ Requirements

Install dependencies:
```bash
pip install ultralytics opencv-python numpy



---

a. modify the video path
b. Run the Program :
     python fall_detection.py