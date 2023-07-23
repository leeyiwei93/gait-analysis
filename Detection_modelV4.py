import mediapipe as mp
import cv2 as cv
import matplotlib.animation as animation
import numpy as np
import math
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

colors = [(245,117,16), (117,245,16), (16,117,245)]
time = 0

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
        cv.putText(output_frame, actions[num], (0, 85+num*40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
    return output_frame

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def calculateAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians =  np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0]- b[0])
    angle = np.abs ( radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360 - angle
        
    return angle

actions = np.array(['right_heel_strike', 'right_toe_off'])

BG_COLOR = (192, 192, 192)
predictions = []
sequence = []
sentence = []
walk_type =[]
angle =[]
threshold = 0.9
walk_flag = 0
window_size = 3
feet_angle = "No stance detected"
actions = np.array(['right_heel_strike', 'right_toe_off'])


with open('gaitanalysis.pkl', 'rb') as f:
    model = pickle.load(f)

#class_name = "sit spin"
#VIDEO FEED with mediapipe pose estimation
cap = cv.VideoCapture('gait_python/walk_mid_ns.mp4') #0 for camera
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks= True, enable_segmentation=True,
model_complexity=2) as pose:
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame = cv.flip(frame,0)
            frame = cv.flip(frame, 1)
            height, width, _ = frame.shape
            width = int(frame.shape[1] * 40 / 100)
            height = int(frame.shape[0] * 40 / 100)
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
            annotated_image, results = mediapipe_detection(annotated_image, pose)

            #extract the keypoints points
            try:
                landmarks = results.pose_landmarks.landmark
                for i in range(len(landmarks)):
                 poses = results.pose_landmarks.landmark
                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in poses]).flatten())if results.pose_landmarks else np.zeros(33*2)
                 del pose_row[0:66]
               
                RHEEL_coor = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y])
                RBigToe_coor = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])

                X_feet = (RHEEL_coor[0]-RBigToe_coor[0])*frame.shape[1]
                Y_feet = (RHEEL_coor[1]-RBigToe_coor[1])*frame.shape[0]
                
                sequence.append(pose_row)
                sequence = sequence[-30:]

                X = pd.DataFrame([pose_row])
                # print(X[0])
                body_pose = model.predict(X)
                body_pose_prob = model.predict_proba(X)[0] #outputs the probability of the 2 classes in an array[x,y,z]
                
                print((np.argmax(body_pose_prob)))
                if len(sequence) == 30:
                    predictions.append(np.argmax(body_pose_prob))

                    #3. Viz logic
                    if body_pose_prob[np.argmax(body_pose_prob)] > threshold:
                        if (np.argmax(body_pose_prob) == 0):              
                            if len(sentence) > 0:
                                if actions[np.argmax(body_pose_prob)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(body_pose_prob)])
                            else:
                                sentence.append(actions[np.argmax(body_pose_prob)])
                        elif (np.argmax(body_pose_prob) == 1): 
                            if len(sentence) > 0:
                                if actions[np.argmax(body_pose_prob)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(body_pose_prob)])
                            else:
                                sentence.append(actions[np.argmax(body_pose_prob)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                    
                    if len(sentence) > 0:
                        walk_type = sentence[-1:][0]

                image = prob_viz(body_pose_prob, actions, image, colors)
                

                cv.rectangle(image, (0,0), (640, 40), (0,255,0), -1)
                cv.rectangle(image, (900,0), (1300,100), (0,255,0), thickness = -1)
                cv.putText(image, str(sentence[-1:]), (3,30), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)


                if walk_type == 'right_heel_strike' and walk_flag != 1:
                    feet_angle =  abs(math.degrees((math.atan(Y_feet/X_feet))))
                    walk_flag = 1
                    print(feet_angle)

                elif walk_type == 'right_toe_off' and walk_flag != 0:
                    feet_angle =  abs(math.degrees((math.atan(Y_feet/X_feet))))
                    walk_flag = 0
                    print(feet_angle)


                
                cv.putText(image, "feet_angle " + str(round(feet_angle)), (0,600), 
                cv.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,225), thickness = 1)
                            
            except:
                pass
            

        except:
            pass    
        
        h, w = image.shape[0:2]
        neww = 1000
        newh = int(neww*(h/w))
        print(width, height)
        img = cv.resize(image, (width, height))

        cv.imshow('Video', img)
    
        

        if cv.waitKey(10) & 0xFF == ord('q'): #if the q key is pressed stop the video
                    break

    cap.release()
    cv.destroyAllWindows