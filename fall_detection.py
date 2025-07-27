


# import cv2
# import numpy as np
# import math
# from ultralytics import YOLO
# import torch

# # YOLOv8 pose model
# model = YOLO('yolov8n-pose.pt')
# model.to('cuda' if torch.cuda.is_available() else 'cpu')

# def fall_detection_yolov8(keypoints, image_height, image_width):
#     """Detects if a person has fallen using YOLOv8 pose keypoints."""
#     if keypoints is None or len(keypoints) < 17:  # Check if all keypoints are detected
#         return False, None

#     try:
#         left_shoulder = keypoints[5]
#         right_shoulder = keypoints[6]
#         left_hip = keypoints[11]
#         right_hip = keypoints[12]
#         left_ankle = keypoints[15]
#         right_ankle = keypoints[16]
#         left_foot_index = keypoints[15] #using ankle as foot index, since yolov8 pose doesn't have foot index.
#         right_foot_index = keypoints[16] #using ankle as foot index, since yolov8 pose doesn't have foot index.

#         if not all([left_shoulder[2] > 0.5, right_shoulder[2] > 0.5, left_hip[2] > 0.5, right_hip[2] > 0.5, left_ankle[2] > 0.5, right_ankle[2] > 0.5]): #check confidence.
#             return False, None

#         left_shoulder_y = left_shoulder[1]
#         left_shoulder_x = left_shoulder[0]
#         right_shoulder_y = right_shoulder[1]
#         left_hip_y = left_hip[1]
#         left_hip_x = left_hip[0]
#         right_hip_y = right_hip[1]
#         left_ankle_y = left_ankle[1]
#         right_ankle_y = right_ankle[1]
#         left_foot_y = left_foot_index[1]
#         right_foot_y = right_foot_index[1]

#         len_factor = math.sqrt(((left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_x - left_hip_x) ** 2))

#         # Calculate bounding box
#         all_x = [kp[0] for kp in keypoints if kp[2] > 0.5]
#         all_y = [kp[1] for kp in keypoints if kp[2] > 0.5]

#         if not all_x or not all_y:
#             return False, None

#         xmin, xmax = min(all_x), max(all_x)
#         ymin, ymax = min(all_y), max(all_y)

#         dx = int(xmax) - int(xmin)
#         dy = int(ymax) - int(ymin)
#         difference = dy - dx

#         if (left_shoulder_y > left_foot_y - len_factor and left_hip_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_hip_y - (len_factor / 2)) or \
#            (right_shoulder_y > right_foot_y - len_factor and right_hip_y > right_foot_y - (len_factor / 2) and right_shoulder_y > right_hip_y - (len_factor / 2)) or \
#            difference < 0:
#             return True, (xmin, ymin, xmax, ymax)
#         return False, None

#     except IndexError:
#         return False, None

# def falling_alarm(image, bbox, keypoints):
#     x_min, y_min, x_max, y_max = bbox

#     # Calculate text position above the person's head
#     text_x = int((x_min + x_max) / 2)
#     text_y = int(y_min - 20)  # Adjust the offset as needed

#     cv2.putText(image, 'Person Fell down', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)

# def process_video_yolov8(video_source):
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Error: Could not open video source.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.predict(frame)

#         for result in results:
#             if result.keypoints:
#                 keypoints_list = result.keypoints.data.cpu().numpy() #get all keypoints.
#                 boxes = result.boxes.xyxy.cpu().numpy()

#                 for i, keypoints in enumerate(keypoints_list): #loop through each person.
#                     image_height, image_width, _ = frame.shape
#                     is_fall, bbox = fall_detection_yolov8(keypoints, image_height, image_width)
#                     print(f"is_fall: {is_fall}")

#                     if is_fall:
#                         falling_alarm(frame, bbox, keypoints)

#                     # Draw keypoints (Red, smaller dots)
#                     for kp in keypoints:
#                         if kp[2] > 0.5: #only draw keypoints if confidence > 0.5.
#                             cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)  #Red, radius 3

#             # Draw bounding boxes (optional)
#             if result.boxes:
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 for box in boxes:
#                     x1, y1, x2, y2 = map(int, box)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

#         cv2.imshow('YOLOv8 Pose Fall Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     #To use live camera
#     #video_source = 0 # 0 for default camera
#     #To use file path.
#     video_source = 'video_6.mp4'  # Replace with your video file path
#     process_video_yolov8(video_source)








####### saving output

import cv2
import numpy as np
import math
from ultralytics import YOLO
#import torch

# YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')
# model.to('cuda' if torch.cuda.is_available() else 'cpu')

def fall_detection_yolov8(keypoints, image_height, image_width):
    """Detects if a person has fallen using YOLOv8 pose keypoints."""
    if keypoints is None or len(keypoints) < 17:  # Check if all keypoints are detected
        return False, None

    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        left_foot_index = keypoints[15] #using ankle as foot index, since yolov8 pose doesn't have foot index.
        right_foot_index = keypoints[16] #using ankle as foot index, since yolov8 pose doesn't have foot index.

        if not all([left_shoulder[2] > 0.5, right_shoulder[2] > 0.5, left_hip[2] > 0.5, right_hip[2] > 0.5, left_ankle[2] > 0.5, right_ankle[2] > 0.5]): #check confidence.
            return False, None

        left_shoulder_y = left_shoulder[1]
        left_shoulder_x = left_shoulder[0]
        right_shoulder_y = right_shoulder[1]
        left_hip_y = left_hip[1]
        left_hip_x = left_hip[0]
        right_hip_y = right_hip[1]
        left_ankle_y = left_ankle[1]
        right_ankle_y = right_ankle[1]
        left_foot_y = left_foot_index[1]
        right_foot_y = right_foot_index[1]

        len_factor = math.sqrt(((left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_x - left_hip_x) ** 2))

        # Calculate bounding box
        all_x = [kp[0] for kp in keypoints if kp[2] > 0.5]
        all_y = [kp[1] for kp in keypoints if kp[2] > 0.5]

        if not all_x or not all_y:
            return False, None

        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)

        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx

        if (left_shoulder_y > left_foot_y - len_factor and left_hip_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_hip_y - (len_factor / 2)) or \
           (right_shoulder_y > right_foot_y - len_factor and right_hip_y > right_foot_y - (len_factor / 2) and right_shoulder_y > right_hip_y - (len_factor / 2)) or \
           difference < 0:
            return True, (xmin, ymin, xmax, ymax)
        return False, None

    except IndexError:
        return False, None

def falling_alarm(image, bbox, keypoints):
    x_min, y_min, x_max, y_max = bbox

    # Calculate text position above the person's head
    text_x = int((x_min + x_max) / 2)
    text_y = int(y_min - 20)  # Adjust the offset as needed

    cv2.putText(image, 'Person Fell down', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)

def process_video_yolov8(video_source, output_path='output.avi'):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            if result.keypoints:
                keypoints_list = result.keypoints.data.cpu().numpy() #get all keypoints.
                boxes = result.boxes.xyxy.cpu().numpy()

                for i, keypoints in enumerate(keypoints_list): #loop through each person.
                    image_height, image_width, _ = frame.shape
                    is_fall, bbox = fall_detection_yolov8(keypoints, image_height, image_width)
                    print(f"is_fall: {is_fall}")

                    if is_fall:
                        falling_alarm(frame, bbox, keypoints)

                    # Draw keypoints (Red, smaller dots)
                    for kp in keypoints:
                        if kp[2] > 0.5: #only draw keypoints if confidence > 0.5.
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)  #Red, radius 3

            # Draw bounding boxes (optional)
            if result.boxes:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out.write(frame) #write frame to output.

        cv2.imshow('YOLOv8 Pose Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release() #release video writer.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #To use live camera
    #video_source = 0 # 0 for default camera
    #To use file path.
    video_source = r'C:\Users\admin\Desktop\person_fall_detection\fall_detection video\video_5.mp4'  # Replace with your video file path
    output_path = 'fall_video_output.mp4' #output video path
    process_video_yolov8(video_source, output_path)











