import mediapipe as mp
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import os

#function to calculate angles from 2 vectors 
def vectors_to_angle(vector1, vector2) -> float:
    x = np.dot(vector1, -vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    theta = np.degrees(np.arccos(x))
    return theta

#this function takes coordinates of every frame and computes the angle and append to a dictionary
def landmark_to_angle(lndmks) -> dict:
    
    # coordinates
    Nose_coor = np.array([lndmks[mp_pose.PoseLandmark.NOSE.value].x, lndmks[mp_pose.PoseLandmark.NOSE.value].y])
    LHip_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_HIP.value].x, lndmks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    RHip_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    MidHip_coor = np.array([(LHip_coor[0] + RHip_coor[0])/2, (LHip_coor[1] + RHip_coor[1])/2])
    LKnee_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lndmks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
    RKnee_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
    LAnkle_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lndmks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    RAnkle_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
    LBigToe_coor = np.array([lndmks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, lndmks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])
    RBigToe_coor = np.array([lndmks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, lndmks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
    
    # vectors
    Torso_vector = MidHip_coor - Nose_coor
    Hip_vector = LHip_coor - RHip_coor
    LFemur_vector = LKnee_coor - LHip_coor
    RFemur_vector = RKnee_coor - RHip_coor
    LTibia_vector = LAnkle_coor - LKnee_coor
    RTibia_vector = RAnkle_coor - RKnee_coor
    LFoot_vector = LBigToe_coor - LAnkle_coor
    RFoot_vector = RBigToe_coor - RAnkle_coor
    
    # angles
    TorsoLHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    TorsoRHip_angle = vectors_to_angle(Torso_vector, Hip_vector)
    LHip_angle = vectors_to_angle(LFemur_vector, Hip_vector)
    RHip_angle = vectors_to_angle(RFemur_vector, -Hip_vector)
    LKnee_angle = vectors_to_angle(LTibia_vector, LFemur_vector)
    RKnee_angle = vectors_to_angle(RTibia_vector, RFemur_vector)
    LAnkle_angle = vectors_to_angle(LFoot_vector, LTibia_vector)
    RAnkle_angle = vectors_to_angle(RFoot_vector, RTibia_vector)
    

    dict_angles = {"TorsoLHip_angle": TorsoLHip_angle, "TorsoRHip_angle": TorsoRHip_angle, "LHip_angle": LHip_angle,
                   "RHip_angle": RHip_angle, "LKnee_angle": LKnee_angle, "RKnee_angle": RKnee_angle, "LAnkle_angle": LAnkle_angle, "RAnkle_angle": RAnkle_angle}
    return dict_angles

# #EQUATION TO CALCULATE ANGLE FROM 3 VECTORS
# def calculateAngle(a,b,c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     radians =  np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0]- b[0])
#     angle = np.abs ( radians*180.0/np.pi)

#     if angle >180.0:
#         angle = 360 - angle
        
#     return angle

def plot_angles(df) -> None:
    
    with sns.axes_style("darkgrid"):
        fig, axes = plt.subplots()
        
        fig.suptitle("Changes over time of joint angles")
        
        # sns.lineplot(ax = axes[0, 0], data = df, x = "Time_in_sec", y = "TorsoLHip_angle").set(xlabel = "Time (seconds)", ylabel = "Torso with left hip (º)")
        # sns.lineplot(ax = axes[0, 1], data = df, x = "Time_in_sec", y = "TorsoRHip_angle").set(xlabel = "Time (seconds)", ylabel = "Torso with right hip (º)")
        # sns.lineplot(ax = axes[1, 0], data = df, x = "Time_in_sec", y = "LHip_angle").set(xlabel = "Time (seconds)", ylabel = "Hip with left leg (º)")
        # sns.lineplot(ax = axes[1, 1], data = df, x = "Time_in_sec", y = "RHip_angle").set(xlabel = "Time (seconds)", ylabel = "Hip with right leg (º)")
        # sns.lineplot(data = df, x = "Time_in_sec", y = "LKnee_angle").set(xlabel = "Time (seconds)", ylabel = "Left knee (º)")
        sns.lineplot(data = df, x = "Time_in_sec", y = "RKnee_angle").set(xlabel = "Time (seconds)", ylabel = "Right knee (º)")
        # sns.lineplot(ax = axes[2, 1], data = df, x = "Time_in_sec", y = "LAnkle_angle").set(xlabel = "Time (seconds)", ylabel = "Left ankle (º)")
        # sns.lineplot(ax = axes[2, 2], data = df, x = "Time_in_sec", y = "RAnkle_angle").set(xlabel = "Time (seconds)", ylabel = "Right ankle (º)")
        
        plt.show()



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
fr = 30
list_of_dicts = []
framenumber = 0
framecount = []
fig = plt.figure()

#file = os.path.join('Videos', 'Sit_spin_test_1.mp4')#'D:\Python\mediapipe\VideosforTesting\skating.mp4' ***************

cap = cv.VideoCapture("walk_test2.mp4") #0 for camera ****************
with mp_pose.Pose(min_detection_confidence = 0.4, min_tracking_confidence = 0.4, model_complexity=1) as pose:

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            frame = cv.flip(frame,0)
            height, width, _ = frame.shape
            #recolor to rgb
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            #make detection
            results = pose.process(image)

            #return image color to bgr
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
            #rendering the detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color = (0,0,255), thickness = 2, circle_radius = 1),
            mp_drawing.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2))
        
        
            try:
                landmarks = results.pose_landmarks.landmark
                list_of_dicts.append(landmark_to_angle(landmarks))
                df_angles = pd.DataFrame(list_of_dicts)
                framenumber += 1
                framecount.append(framenumber)
                
            except:
                pass


            # h, w = image.shape[0:2]
            # neww = 1000
            # newh = int(neww*(h/w))
            # converts image/video resolution according to the scale percent
            scale_percent = 40 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            img = cv.resize(image, (width, height))
           
            cv.imshow('Video', img)

            if cv.waitKey(30) & 0xFF == ord('q'): #if the q key is pressed stop the video
                break
        except:
            break

cap.release()
cv.destroyAllWindows()

df_angles = pd.DataFrame(list_of_dicts)
df_angles.insert(0, "Frame", framecount, True)
print(df_angles)
# df_angles["Time_in_sec"] = [n/fr for n in range(len(df_angles))]

# df_to_plot = df_angles.loc[(df_angles["Time_in_sec"] >= 0) & (df_angles["Time_in_sec"] <= 5)]

# plot_angles(df_to_plot)
# x = df_angles.Frame
# y = df_angles.RKnee_angle
# plt.plot(x, y)
# plt.show()