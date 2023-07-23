import mediapipe as mp
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

BG_COLOR = (192, 192, 192)

file1 = os.path.join('gaitData2.csv')#'D:\Python\mediapipe\SpinData.csv' ********


#check if file exist, if not create********
if os.path.isfile(file1):
    pass
else:
    num_coords = (11)

    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

    with open(file1, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
    

class_name = "right_toe_off" #change class according to training video
#VIDEO FEED with mediapipe pose estimation

file = os.path.join('gait_python', 'Rtoe_off1.mp4')#'D:\Python\mediapipe\VideosforData\camel_spin#.mov'**********

cap = cv.VideoCapture(file) #0 for camera ********
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks= True, enable_segmentation=True, 
model_complexity=2) as pose:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            # frame = cv.flip(frame,0)
            # frame = cv.flip(frame, 1)
            height, width, _ = frame.shape
            width = int(frame.shape[1] * 60 / 100)
            height = int(frame.shape[0] * 60/ 100)
            frame = cv.resize(frame, (width, height))

            # Make detections
            image, results = mediapipe_detection(frame, pose)

            # Make segmentation
            annotated_image = image

            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)

            #make detections again after segmentation
            image, results = mediapipe_detection(annotated_image, pose)


            #extract the keypoints points
            try:
                landmarks = results.pose_landmarks.landmark
                for i in range(len(landmarks)):
                 poses = results.pose_landmarks.landmark
                 pose_row = list(np.array([[landmark.x, landmark.y,landmark.z] for landmark in poses]).flatten())if results.pose_landmarks else np.zeros(33*2)
                 del pose_row[0:66]
                 print(pose_row)
                
                # Append class name 
                pose_row.insert(0, class_name)
                # Export to CSV
                with open(file1, mode='a', newline='') as f: #***********
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(pose_row) 
            except:
                pass
            


            #Resize ********
            h, w = image.shape[0:2]
            neww = 1000
            newh = int(neww*(h/w))
            img = cv.resize(image, (width, height))

        except:
            pass

        cv.imshow('Video', img)# **********
        if cv.waitKey(30) & 0xFF == ord('d'): #if the d key is pressed stop the video
            break

    cap.release()
    cv.destroyAllWindows


