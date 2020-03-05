import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
from inference.inference_general_utils import *

class FaceAnalyzer:
    def __init__(self,
                 age_net,
                 aff_net, 
                 pose_net):

        self.age_net = age_net
        self.aff_net = aff_net
        self.pose_net = pose_net
        

    def parse_age_gender(self, inference):
        '''
        this function gets the output of a inference of the age_gender_net and gets its label as male of female
        '''
        age = np.round(100 * float(inference["age_conv3"]))
        if np.argmax(inference["prob"]) == 1:
            gender = "male"
        else:
            gender = "female"
        return age, gender

    def parse_sentiment(self, inference):
        '''
        this function gets the output of a inference of the age_gender_net and gets its label as the face expression recognized
        '''
        sentiments = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        label = np.argmax(inference['prob_emotion'])
        return sentiments[label]

    def infere_from_face(self,
                         frame,
                         gray,
                         face):
        '''
        this function runs a face troughout age, gender and head-pose net and aff net, both plugged in with OpenVINO Inference Engine
        
        parameters:
        frame: matrix nxmx3, the latest captured by the camera
        gray: matrix nxmx1, the same frame, but in gray-scale, so it runs faster
        face: (x, y, width and height), coordinates to find face
        age-aff nets: OpenVINO Inference Engine Executable network
        
        what happens here is that we select subset the frame matrix to get only the person's face, so that we can resize it and run
        it through some neural networks and get its labels
        
        as the frame is passed by value due to Python, I decided to use this function to actually plot it in the frame, so we can see
        the detections
        '''
        
        (x, y, w, h) = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        (startX, startY, endX, endY) = (x, y, x+w, y+h)
        age_inf = self.age_net.infere_from_image(roi_gray)
        age_label = self.parse_age_gender(age_inf)

        aff_inf = self.aff_net.infere_from_image(roi_gray) #infere_from_image(roi_gray, (64,64), aff_net)
        aff_label = self.parse_sentiment(aff_inf)
        
        pose_inf = self.pose_net.infere_from_image(roi_color) #infere_from_image(roi_color, (60,60), pose_net)
        (yaw, pitch, roll) = pose_inf['angle_y_fc'][0][0], pose_inf['angle_p_fc'][0][0], pose_inf['angle_r_fc'][0][0]
        
        cv2.putText(frame, "Yaw: " + str(yaw), (x, y -50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(frame, "Pitch: " + str(pitch), (x, y-35),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, "Roll: "+str(roll), (x, y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        label = str(age_label) + str(aff_label) 
        cv2.putText(frame, str(label), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        return (age_label, aff_label), (yaw, pitch, roll)
