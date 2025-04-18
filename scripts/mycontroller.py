#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

import math
import threading

class GestureController(Node):
    def __init__(self):
        super().__init__('Gesture_Control')

        # Publishers for single Arm
        
        self.armcontroller = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Initialize MediaPipe hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands()

        # Initialize webcam input
        self.cap = cv2.VideoCapture(0)

        # Run ROS2 processing in a separate thread
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self,))
        self.ros_thread.start()


    def process_gestures(self):
        alpha = 0.2 # Transparency factor for overlaid images
        hf_center_x=130
        hf_center_y=360
        hf_right_shift=370
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            overlay=frame.copy()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            HandNum=0
            # AHand detection logic
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # rl_value: 1 => right, 2 => left
                    rl_value = results.multi_handedness[HandNum].classification[0].index 
                    # Use hand position or gesture classification to set twist_right or twist_left
                    center_x, center_y = get_hand_center(hand_landmarks, frame.shape[1], frame.shape[0])

                    hand_center=(center_x, center_y)
                    bullseye_center=(hf_center_x+hf_right_shift*rl_value, hf_center_y)
                    bullseye_radii=(40,100)
                    gesture = recognize_gesture(hand_landmarks)

                    tilt = recognize_handtilt(hand_landmarks)

                    relative_pos=(hand_center[0]-bullseye_center[0],hand_center[1]-bullseye_center[1], tilt)
                    
                    if rl_value==1:
                        twist_command = twist_calculation(gesture,relative_pos,bullseye_radii)
                        twist_command.header.stamp = self.get_clock().now().to_msg()
                        self.armcontroller.publish(twist_command)
                    else:
                        twist_command = twist_calculation(gesture,relative_pos,bullseye_radii)
                        twist_command.header.stamp = self.get_clock().now().to_msg()
                        self.armcontroller.publish(twist_command)
                    
                    #print("Twist: ")
                    #print(twist_command)
                    #print(TwistStamped)

                    
                    ##############################
                    # Drawing on the camera view #
                    side_text="Right" if rl_value==1 else "Left"
                    textLocx=50 +300*rl_value
                    
                    self.mp_drawing.draw_landmarks(overlay, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    cv2.putText(frame, side_text, (textLocx, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Gesture: {gesture}", (textLocx, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    #cv2.putText(frame, f"Center: ({center_x}, {center_y})", (textLocx, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Disp: {relative_pos}", (textLocx, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    HandColor=(255*(1-rl_value), 0, 255*rl_value)

                    cv2.circle(overlay, hand_center, 20, HandColor, -1)
                    

                    # Small circle around center
                    cv2.circle(overlay, bullseye_center, bullseye_radii[0] , HandColor, 5)

                    # Big circle around center
                    cv2.circle(overlay, bullseye_center, bullseye_radii[1] , HandColor, 5)

                    cv2.line(overlay,bullseye_center,hand_center,HandColor,5)

                    # Following line overlays transparent rectangle 
                    #  over the image 

                    # If the other hand is 
                    HandNum=HandNum+1
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0) 

            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    print('Hi from GestureControlSingleArm.')
    rclpy.init()
    mynode = GestureController()
    mynode.process_gestures()

    mynode.destroy_node()
    rclpy.shutdown()

def get_hand_center(landmarks, frame_width, frame_height):
    palmlandmarks=[0, 5, 17]

    x_coords = [landmarks.landmark[i].x * frame_width for i in palmlandmarks]
    y_coords = [landmarks.landmark[i].y * frame_height for i in palmlandmarks]

    return int(np.mean(x_coords)), int(np.mean(y_coords))

def recognize_gesture(landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]
    
    for i in range(0, 5):
        fingers.append(is_finger_open(tips[i], landmarks))
    
    # Gesture dictionary
    gestures = {
        (0, 0, 0, 0, 0): "Fist",
        (0, 1, 1, 0, 0): "Peace Sign",
        (1, 1, 1, 1, 1): "Open Palm",
        (0, 1, 0, 0, 0): "Index Finger",
        (1, 0, 0, 0, 0): "Thumbs Up",
        (0, 1, 1, 1, 1): "Four Fingers",
        (0, 0, 1, 1, 1): "Three Fingers",
        (0, 0, 0, 1, 1): "Two Fingers"
    }

    #print(fingers)
    return gestures.get(tuple(fingers), "Unknown Gesture")

def recognize_handtilt(landmarks):
    # Calculating a simplistic hand tilt parameter
    handrootx=landmarks.landmark[0].x 
    middlefingerrootx=landmarks.landmark[13].x
    # middlefingerrootx=(landmarks.landmark[9].x +landmarks.landmark[13].x )/2
    tilt= middlefingerrootx-handrootx
    scale=-10
    return round(tilt*scale,2)

def is_finger_open(fingertipind, landmarks):
    # Check if finger is open

    # If checking thumb
    tip2root=2 if fingertipind==4 else 3

    handroot = landmarks.landmark[0]
    fingertip = landmarks.landmark[fingertipind]
    fingerroot = landmarks.landmark[fingertipind-tip2root]

    ftLoc=np.array([fingertip.x, fingertip.y])
    frLoc=np.array([fingerroot.x, fingerroot.y])
    hrLoc=np.array([handroot.x, handroot.y])

    # Calculate distance vectors between wrist to knuckle and knuckle to fingertip
    vector1 = ftLoc - frLoc
    vector2 = frLoc - hrLoc

    uv1 = vector1/np.linalg.norm(vector1)
    uv2 = vector2/np.linalg.norm(vector2)

    # Calculate the dot product of the vectors
    r_finger= np.dot(uv1, uv2)

    if r_finger>0.6:
    #if landmarks.landmark[tips[0]].x > landmarks.landmark[tips[0] - 2].x:
        return 1  # Finger extended
    else:
        return 0
    

def twist_calculation(gesture,hand_relative_pos,bullseye_radii):
    twist_command=TwistStamped()
    # twist_command.header.frame_id = "panda_link0"
    twist_command.header.frame_id = "base"
    twist_command.twist.linear.x = 0.0
    twist_command.twist.linear.y = 0.0
    twist_command.twist.angular.z = 0.0
    scale_xy=0.01
    rx=hand_relative_pos[0]*scale_xy
    ry=hand_relative_pos[1]*scale_xy*-1
    rz=hand_relative_pos[2]
    theta=0.0

    rx=Saturation(rx,1)
    ry=Saturation(ry,1)
    rz=Saturation(rz,1)

    match gesture:
        case "Peace Sign":
            twist_command.twist.angular.z = rz
        case "Index Finger": # Position mode
            twist_command.twist.linear.x = rx*math.cos(theta) + ry*math.sin(theta)
            twist_command.twist.linear.y = -rx*math.sin(theta) + ry*math.cos(theta)
        case "Fist": # Go straight
            Cartesianx=rx*math.cos(theta) + ry*math.sin(theta)
            Cartesiany=rx*math.sin(theta) - ry*math.cos(theta)
            twist_command.twist.linear.y = Cartesianx
            twist_command.twist.linear.x = Cartesiany
        case "Open Palm": # Stop
            twist_command.twist.linear.x = 0.0
    return twist_command

def Saturation(a,maxNum):
    if a>maxNum:
        return maxNum
    elif a<-maxNum:
        return -maxNum
    else: 
        return a

if __name__ == '__main__':
    main()


