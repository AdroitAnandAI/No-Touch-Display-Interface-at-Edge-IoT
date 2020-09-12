# No-Touch Display Interface at Edge (IoT)
## _Enabling HCI on Edge: Multi-threaded Gesture & Sound Control of kiosks with Intel OpenVINO AI models. Eye Wink & Mouth Aspect with numerical models_

The objective of the project is to detect faces in the video from file or cam. Further, the eyes are detected and head pose orientation is computed using a pre-trained openvino model. The results of the models are cascadingly fed to a gaze estimation model which predicts the angle of gaze of the person. The angle of gaze is taken as input and used to control the mouse pointer of the screen.

The output of the project depends on the intial calibration step where the system recognized the extremities of the screen and corresponding gaze angles.

IMPORTANT: 
If the mouse pointer is behaving incorrectly then the problem is in the calibration step. Please make sure that your face is properly lighted and positioned approximately to middle of the screen so that the gaze angles would make sense of left, right, top and bottom.

## Project Set Up and Installation

Directory Structure:

├── src                     # All components: face, gaze, headpose, landmarks and mouse_controller
   ├──noTouchKiosk.py         # The main file to execute: python3 noTouchKiosk.py
├── images                  # Required images
├── bin                     # Demo video taken as input
├── models                  # All model files downloaded here
├── requirements.txt        # Project Dependancies
└── README.md


# Install OpenVino
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
tar -xvf l_openvino_toolkit_p_2020.2.120.tgz
cd l_openvino_toolkit_p_2020.2.120
sed -i 's/decline/accept/g' silent.cfg
sudo ./install.sh -s silent.cfg

# Create a Virtual env
python3 -m venv edge
source edge/bin/activate

# Project dependancies
pip3 install -r requirements.txt

To download the required models:
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "facial-landmarks-35-adas-0002"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-0001"


## Demo

How To run:
python3 noTouchKiosk.py {command line arguments}

Example: python3 noTouchKiosk.py -f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True

You can change the model precision and flags given as parameters. vh, -vg and -vf are the visualiation debug flags.

The code allows the user to set a flag that can display the outputs of intermediate models. For instance, -vh to visualize head pose results and -vg to visualize gaze model outputs.

IMPORTANT: 
The output of the project depends on the intial calibration step where the system recognized the extremities of the screen and corresponding gaze angles. Upon execution, the system will direct you to look at the top right corner and then the bottom left corner of your computer screen. Based on the corresponding gaze angles the system is capable to compute the intermediate x, y coordinates using interpolation techniques.

Note: The calibration step is optimized for cam input where the user can look at the screen corners. If video is given as input then the mouse will be controlled according to gaze but the direction of mouse pointer can differ. This happens because the person in the video is not looking at the corner of the screen.

On a different note, if you visualize the output of head pose model then it gives angle of vision but then the orientation of eye balls are not taken into consideration. Instead, when the eyes are cropped and fed into gaze estimation model, the angle of sight is correctly estimated, considering both head pose as well as location of eye ball.

## Documentation

Command Line Arguments:
-f:  Path to .xml file of Face Detection model
-l:  Path to .xml file of Facial Landmark Detection model
-hp: Path to .xml file of Head Pose Estimation model
-ge: Path to .xml file of Gaze Estimation model
-i:  Path to video file or enter cam for webcam
-it: Provide the source of video frames
-d:  Provide the target device: "CPU, GPU, FPGA or MYRIAD is acceptable."

I have used the heavy face landmark detection model - facial-landmarks-35-adas-0002 - which can detect 35 facial landmarks, instead of landmarks-regression-retail-0009 which can detect just 5 facial features. The reason is that, a human computer interface controller which can control and give commands with eye and mouth gestures requires dense landmark points. For instance, left eye wink for left mouse click and right eye wink for right click. 

The above requirement needs estimation of more facial features in maximum detail and hence the change in model. The current model - facial-landmarks-35-adas-0002 - demands 0.042 GFlops in place of 0.021 GFlops reqired by landmarks-regression-retail-0009.

The Euler angles are converted to cartesian coordinates using rotation matrix values:

Rotation matrix is:

| cos(yaw)cos(pitch) -cos(yaw)sin(pitch)sin(roll)-sin(yaw)cos(roll) -cos(yaw)sin(pitch)cos(roll)+sin(yaw)sin(roll)|
| sin(yaw)cos(pitch) -sin(yaw)sin(pitch)sin(roll)+cos(yaw)cos(roll) -sin(yaw)sin(pitch)cos(roll)-cos(yaw)sin(roll)|
| sin(pitch)          cos(pitch)sin(roll)                            cos(pitch)sin(roll)|



## Benchmarks

Used the below parameters for corresponding benchmarks:

	# To parse the video file given - all FP16 models
     arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it video -d CPU -vh False -vg True -vf True'.split(' ')
    
    # To take input from the webcam - all FP16 models
     arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True'.split(' ')

    # To take input from the webcam but with FP32 gaze & Landmark detection models
     arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True'.split(' ')


    # To take input from webcam but with INT8 Face detection & FP32 gaze & Landmark detection models
     arg = '-f ../models/face-detection-adas-0001/FP32-INT8/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True'.split(' ')


## Results


a) Benchmark by taking video input. All models are FP16
FPS = 15.29160590328414
Inference Time of 4 models = 0.08967375755310059

b) Benchmark by taking webcam as input. All models are FP16
Inference Time of 4 models = 0.031467437744140625
FPS = 17.09894984019307

c) Webcam as input. Using FP16 for Face Detection & Head pose and FP32 for Gaze & Landmark detection models
Inference Time of 4 models = 0.04231882095336914
FPS = 14.50613543612091

d) Webcam as input. Using INT8 for Face Detection and remaining are FP16 models
Inference Time of 4 models = 0.028314828872680664
FPS = 18.337839492138997

e) Webcam as input. Using INT8 for Face Detection & FP32 for Gaze & Landmark detection models and FP16 for head pose model.

Inference Time of 4 models = 0.03919792175292969
FPS = 15.08791291804411


It is seemingly clear that increase in number of bits slows down the inference and lower the FPS. But as for all AI tasks, its a tradeoff between required accuracy and minimum speed.

For face detection, INT8 model is giving good accuracy but landmark detection and gaze require maximum accuracy for accurate mouse control. Headpose require reasonable accuracy, hence FP16 is used. 

FP16 models commonly regarded as the mid-path between accuracy and speed, as compared to FP32 and INT8. But from the above analysis, it seems like Benchmark (e) gives good balance between accuracy and speed for this project pipeline. 

This project is a human computer interface interface (HCI) which can control and give commands with head, eye, mouth gestures and also sound.
