 # 🖐️ Gesture-Controlled Panda Arm with ROS2 Humble

This ROS2 package allows you to control a panda arm using **hand gestures** via a webcam.

## 📌 Features
- Uses **MediaPipe Hands** for real-time gesture recognition.
- Controls a panda arm based on hand gestures.

---

## 🛠️ Installation & Setup
### 1️⃣ Install ROS2 Humble & Dependencies
Ensure you have **ROS2 Humble** installed, franka and moveit2 packages cloned 
```bash
sudo apt update
git clone https://github.com/moveit/moveit2.git -b $ROS_DISTRO
for repo in moveit2/moveit2.repos $(f="moveit2/moveit2_$ROS_DISTRO.repos"; test -r $f && echo $f); do vcs import < "$repo"; done
rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
```

### 2️⃣ Install Required Python Packages
```bash
pip install mediapipe opencv-python numpy
```

### 3️⃣ Clone & Build the Package
```bash
cd ~/ros2_ws/src
git clone <your-repo-url> gesture_control
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## 🚀 Running the Project
Start the panda arm simulation on Rviz with:
```bash
ros2 launch GestureControlPandaArm teleop_launch.py
```
In a separate terminal window, run the gesture controller 
```bash
ros2 run GestureControlPandaArm mycontroller.py
```

---

## 🖐️ Hand Gestures
| Gesture | Action  |
|---------|--------------------------------------------------|
| Open hand | Stop |
| Fist | Velocity Mode |
| Index finger open | Position Mode (Not complete)  |
| Peace Sign | Rotation Mode |

**Position Mode:** Hand position is in xy-plane translated to panda arm end effector position in xy-plane. (Not fully implemented)

**Velocity Mode:** Hand position is in xy-plane translated to panda arm end effector velocity in xy-plane.

**Rotation Mode:** Hand rotation is translated to panda arm end effector rotation speed. 

**Press 'Q' to quit the gesture recognition window.**


## 📜 License
This project is open-source under the **Apache 2.0**.
