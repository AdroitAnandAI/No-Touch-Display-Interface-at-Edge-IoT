
# Loading common, OS, Computer Vision, argument parser libraries 
import cv2
import imutils
import math
import os
import sys
import linecache
import traceback
import operator
import distutils 
import distutils.util
import logging
import time
import collections 
import numpy as np

import matplotlib.pyplot as plt
from input_feeder import InputFeeder
from argparse import ArgumentParser

# For Signal Processing.
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import peakutils.peak

# Loading all the 4 OpenVino Models
from facedetect import FaceDetector
from headpose import HeadPoseDetect
from landmarks import LandmarksDetect
from gazedetect import GazeDetect
from mouse_controller import MouseController


# For Inter-Thread Communication.
from queue import Queue 
from threading import Thread, Event

# For Speech Recognition (STT)
from audio import audio_helper
from speech_library.speech_manager import SpeechManager
from speech_library.speech_proxy import SPEECH_CONFIG



# LOG_DIR = os.path.join(os.path.pardir, 'log')
MODELS_DIR = os.path.join(os.path.pardir, 'models')


def stopStream(stream_reader):

    if stream_reader:
        stream_reader.stop_stream()
        stream_reader = None

def received_frames(frames, speech, stt):

    speech.push_data(frames, finish_processing=False)
    utt_text, is_stable = speech.get_result()
    rh_result = utt_text.decode("utf-8").strip().lower()

    if (len(rh_result) > 0):
        # print('inside thread: ' + rh_result)
        stt.put(rh_result)



def load_device():
    """Reload audio device list"""
    device_list, default_input_index, loopback_index = \
                                audio_helper.get_input_device_list()

    # indices = [loopback_index, default_input_index] + [0] * (len(self._device_controls)-2)
    # for i, controller in enumerate(self._device_controls):
    #     controller.reload_device_list(device_list, indices[i])
    if not device_list:
        print("No audio devices available")

    return device_list

def detectSoundEvent(utterance, controls, control_syn):

    utters = utterance.split(' ')[-3:]
    utters.reverse()
    print(utters)

    for utter in  utters:
        for control in controls:
            synonyms = control_syn.get(control)
            for synonym in synonyms:
                if synonym in utter:
                    print('Event Trigger: ' + control)
                    return control, utters[-1]

    return None, utters[-1]


def isFaceInBounds(headYawPitchBounds, yaw, pitch):

    minBound = headYawPitchBounds[0]
    maxBound = headYawPitchBounds[1]

    if yaw >= minBound and yaw <= maxBound and \
         pitch >= minBound and pitch <= maxBound:

        return True
    else:
        return False


def sigmoid(x, L ,x0, k, b):
    # print(k)
    # print(np.exp(-k*(x-x0)))
    y = L / (1 + np.exp(k*(x-x0)))+b
    return (y)


def isCurveSigmoid(pixelCounts, count):

    try:
        xIndex = len(pixelCounts)
        # y = list(range(xIndex))
        # print(y)

        p0 = [max(pixelCounts), np.median(xIndex),1,min(pixelCounts)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, list(range(xIndex)), pixelCounts, p0, method='dogbox', maxfev=5000)

        # popt, pcov = curve_fit(sigmoid, list(range(xIndex)), pixelCounts)

        # plt.plot(list(range(xIndex)), sigmoid(list(range(xIndex)), *popt), 'r-', label='fit')
        # plt.show()

        yVals = sigmoid(list(range(xIndex)), *popt)

        # medianY = np.median(yVals)

        print("Pixel Count Size = " + str(xIndex))
        # print("Median Diff = " + str(np.median(yVals[:10]) - np.median(yVals[-10:])))
        # May have to check for a value much less than Median to avoid false positives.
        if np.median(yVals[:10]) - np.median(yVals[-10:]) > 15:
            print('Triggered Event')
            print(yVals)
            xVals = [n+count-40 for n in list(range(xIndex))]
            plt.plot(xVals, sigmoid(list(range(xIndex)), *popt), 'b--', label='Curve Fit')
            # plt.legend()
            plt.pause(1.5)
            return True

    except Exception as err:
        print(traceback.format_exc())
        # PrintException()

    return False



def checkEvent(image, pixelCount, frame_count, numFrames = 50):

    triggerEvent = False

    if (len(image) == 0):
        return pixelCount, False

    # Convert to gray scale as histogram works well on 256 values.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate frequency of pixels in range 0-255 
    histg = cv2.calcHist([gray],[0],None,[256],[0,256])

    # hack to know whether eye is closed or not.
    # more spread of pixels in a histogram signifies an opened eye
    activePixels = np.count_nonzero(histg)
    pixelCount.append(activePixels)

    if len(pixelCount) > numFrames and frame_count % 15 == 0:

        if isCurveSigmoid(pixelCount[-numFrames+10:], len(pixelCount)):
            print('Event Triggered...')
            pixelCount.clear()
            plt.clf()
            triggerEvent = True


    return pixelCount, triggerEvent

    # # Convert to gray scale as histogram works well on 256 values.
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # calculate frequency of pixels in range 0-255 
    # histg = cv2.calcHist([gray],[0],None,[256],[0,256])

    # # hack to know whether eye is closed or not.
    # # more spread of pixels in a histogram signifies an opened eye
    # activePixels = np.count_nonzero(histg)
    # pixelCount.append(activePixels)

    # # If eyes are open and closed for same amount of time then average was 
    # # enough to find if eyes are open or closed. But since eyes are open 
    # # most of the time, the mid of max and min is taken. This will cancel 
    # # out the difference in  lighting conditions also.
    # pixelMid = (np.max(pixelCount) + np.min(pixelCount)) / 2

    # if (len(pixelCount) > 30 and 
    #     activePixels > pixelMid):

    #     # Eyes are open
    #     isEyeOpen.append(True)
    # else:
    #     # Eyes are closed
    #     isEyeOpen.append(False)


    # if (len(isEyeOpen) > lastFrames):

    #     print(isEyeOpen[-lastFrames:])
    #     # Checking whether all false, i.e. eyes are closed in all frames.
    #     if (np.any(isEyeOpen[-lastFrames:]) == False):
    #         print("EVENT TRIGGERED")
    #         isEyeOpen.clear()
    #         pixelCount.clear()
    #         return pixelCount, isEyeOpen, True

    # return pixelCount, isEyeOpen, False

def imshow(windowname, frame, width=None):

    if width is not None:
        frame = imutils.resize(frame, width=width)

    cv2.imshow(windowname, frame)
    cv2.waitKey(25)

def isMoveSignificant(lastPosition, stickiness, x, y):

    last_x = lastPosition[0]
    last_y = lastPosition[1]

    print("Previous X = " + str(last_x) + ". Previous Y = " + \
        str(last_y) + ". Current X = " + str(x) + ". Current Y = " + str(y))

    if abs(last_x - x) > stickiness or abs(last_y - y) > stickiness:
        return True
    else:
        return False


def isMoveEnabled(lastPosition, stickiness, x, y, lastMovements):
    
    moveEnabled = False

    movement = isMoveSignificant(lastPosition, stickiness, x, y)
    lastMovements.append(movement)
    # checking if all are False, i.e. cursor stopped making
    # significant movements for last 'n' frames.
    if not np.any(lastMovements):
        moveEnabled = False

    if not moveEnabled and movement:
        moveEnabled = True

    return moveEnabled, lastMovements



def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


# Function used for debugging list values to fit curves
def writeList(list2Write):

    with open('pixelsCounts.txt', 'w') as filehandle:
        for listitem in list2Write:
            filehandle.write('%s\n' % listitem)


def argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-l", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Provide the source of video frames.")
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-vh", "--visualizeHeadPose", type=str, default=False,
                        help="To debug headpose model's output visually")
    parser.add_argument("-vg", "--visualizeGaze", type=str, default=False,
                    help="To debug gaze model's output visually")
    parser.add_argument("-vf", "--visualizeFace", type=str, default=False,
                    help="To debug face detection model's output visually")
    return parser




def main(args):


    # Multiple Modes of Control 
    ###########################
    ##  0 = No Control
    ##  1 = Gaze Angle Control
    ##  2 = Head Pose Control
    ##  3 = Sound Control
    ###########################
    controlMode = 0
    modes = ['No Control', 'Gaze Control', 'Head Pose', 'Sound Control']

    ####################
    # Control Commands #
    ####################
    # Left Click = Smile 
    # Right Click = Right Eye Blink
    # Scroll Enabled = Left Eye Blink 
    # Increment Control Modes = Mouth Open
    # 
    # You can dictate text in Sound mode (Control mode = 3)


    #####################################################################  
    # Initializing the Speech Recognition Thread
    #####################################################################  

    controls = ['left', 'right', 'up', 'down']

    control_syn = {}
    for control in controls:
        control_syn.setdefault(control, [])

    # Need to account for similar sounding words as speech recog is on the edge!
    control_syn['left'].extend(['let', 'left', 'light', 'live', 'laugh'])
    control_syn['right'].extend(['right', 'write', 'great', 'fight', 'might', 'ride'])
    control_syn['up'].extend(['up', 'hop', 'hope', 'out'])
    control_syn['down'].extend(['down', 'doubt', 'though'])

    # controls = {}

    device_list = load_device()

    stream_reader = audio_helper.StreamReader(
                        device_list[1][0], received_frames)

    if not stream_reader.initialize():
            print("Failed to initialize Stream Reader")
            speech.close()
            speech = None
            return

    speech = SpeechManager()
    print('speech config = ' + str(SPEECH_CONFIG))
    if not speech.initialize(SPEECH_CONFIG,
            infer_device='CPU',
            batch_size=8):
        print("Failed to initialize ASR recognizer")
        speech.close()
        speech = None
        return

    stt = Queue()
    prevUtterance = ''

    reading_thread = Thread(target=stream_reader.read_stream, \
                        args=(speech, stt), daemon=True)
    reading_thread.start()

    #####################################################################    

    # Fixing 40x40 box as yaw and pitch boundaries to
    # correspond to head turning left and right (yaw)
    # and also moving up and down (pitch)
    headYawPitchBounds = [-10, 10]

    lastGaze = [0, 0]
    lastPose = [0, 0]

    # Set the stickiness value
    stickinessHead = 5
    stickinessGaze = 10

    # init the logger
    logger = logging.getLogger()

    feeder = None
    feeder = InputFeeder(args.input_type, args.input)
    feeder.load_data()

    mc = MouseController("medium", "fast")

    # Loading all the gesture control models viz. face, head and gaze
    face_model = FaceDetector(args.face, args.device, args.cpu_extension)
    # face_model.check_model()
    face_model.load_model()
    logger.info("Face Detection Model Loaded...")

    head_model = HeadPoseDetect(args.headpose, args.device, args.cpu_extension)
    # head_model.check_model()
    head_model.load_model()
    logger.info("Head Pose Detection Model Loaded...")

    landmarks_model = LandmarksDetect(args.landmarks, args.device, args.cpu_extension)
    # landmarks_model.check_model()
    landmarks_model.load_model()
    logger.info("Landmarks Detection Model Loaded...")

    gaze_model = GazeDetect(args.gazeestimation, args.device, args.cpu_extension)
    # gaze_model.check_model()
    gaze_model.load_model()
    logger.info("Gaze Detection Model Loaded...")

    visualizeHeadPose = bool(distutils.util.strtobool(args.visualizeHeadPose))
    visualizeGaze = bool(distutils.util.strtobool(args.visualizeGaze))
    visualizeFace = bool(distutils.util.strtobool(args.visualizeFace))


    pixelCount_leye = [] 
    isEyeOpen_leye = [] 
    pixelCount_reye = [] 
    isEyeOpen_reye = []

    isCalibrated = False
    isSmiling = False
    isMouthOpen = False
    moveEnabled = False

    lastPoses = collections.deque(maxlen=20)
    lastGazes = collections.deque(maxlen=20)

    try:
        frame_count = 0

        for ret, frame in feeder.next_batch():

            ################################################################
            # if any sound is deciphered from the spunned off thread then 
            # check the last 3 words of the utterance for matching control world
            if (stt.qsize() > 0 and controlMode == 3):
                
                utterance = stt.get()
                print("From Parent: " + utterance)
                
                # need to process again only if change in utterance
                if (prevUtterance != utterance):
                    control, lastWord = detectSoundEvent(utterance, controls, control_syn)

                    if control is not None:

                        direction = controls.index(control)
                        mc.moveRelative(direction)
                        
                        # isSoundControl = True
                    else:
                        mc.write(lastWord)

                    prevUtterance = utterance

            ################################################################

            k = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if k == ord('q'):
                break

            if not ret:
                break

            frame_count += 1
            crop_face = None

            # inferenceBegin = time.time()
            crop_face, box = face_model.predict(frame.copy())

            if crop_face is None:
                logger.error("Unable to detect the face.")
                continue

            # print(len(crop_face))
            # Draw the face box
            xmin, ymin, xmax, ymax = box
            if visualizeFace:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

            orientation = head_model.predict(crop_face)

            box_left, box_right, \
            left_eye, right_eye, \
            p0, p1, p12, p13, p14, \
            p2, p3, p15, p16, p17, \
            p8, p9, p10, p11 = landmarks_model.predict(crop_face)

            # if any of the eye is not detected eye gesture and 
            # gaze estimation are not executed
            if (left_eye.size * right_eye.size == 0):
                logger.error("Unable to detect eyes.")
                continue

            # print(p0)
            # print(p1)
            pad = 10
            # Compute Right Eye: Close Snap
            right_eye_ball = frame[ymin + p1[1] - pad: ymin + p0[1] + pad,
                                    xmin + p1[0] - pad: xmin + p0[0] + pad]

            # Compute Left Eye: Close Snap
            left_eye_ball = frame[ymin + p3[1] - pad: ymin + p2[1] + pad,
                                    xmin + p2[0] - pad: xmin + p3[0] + pad]


            pixelCount_leye_bk = pixelCount_leye #can delete this line
            pixelCount_reye, Rtrigger = checkEvent(right_eye_ball, pixelCount_reye, frame_count) 
            pixelCount_leye, Ltrigger = checkEvent(left_eye_ball, pixelCount_leye, frame_count) 

            # if Ltrigger and len(pixelCount_reye) > 0:
            #     print("ERROR L...")
            # if Rtrigger and len(pixelCount_leye) > 0:
            #     print("ERROR R...")

            plt.plot(pixelCount_reye, 'r-', label='Right Eye')
            plt.plot(pixelCount_leye, 'g-', label='Left Eye')

            if (frame_count == 1 or Ltrigger or Rtrigger):
                print('Legend Display')
                plt.legend()
            plt.pause(0.05)

            # if Ltrigger and Rtrigger:
            #     print('BOTH EYES ARE CLOSED...')
                
            if Ltrigger:
                print('left eye pressed')
                writeList(pixelCount_leye_bk)
                mc.scroll(20) # you can pass the head pose up/down as param
                # mc.drag()
            elif Rtrigger:
                print('right eye pressed')
                
                mc.clickRight()
            
            # print('eye ball shape = ')
            # print(right_eye_ball.shape)

            
            # calculate frequency of pixels in range 0-255 
            # histg = cv2.calcHist([right_eye_ball],[0],None,[256],[0,256])
            # show the plotting graph of an image 
            # plt.plot(histg) 
            # plt.show()

            # gray = cv2.cvtColor(right_eye_ball, cv2.COLOR_BGR2GRAY) 
            # edged = cv2.Canny(gray, 30, 200) 
            # contours, _ = cv2.findContours(edged,  
            #        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
            # cv2.drawContours(right_eye_ball, contours, -1, (0, 255, 0), 3)
            # imshow('iframe', histg)
            # cv2.imwrite("../eyeImages/eye"+str(frame_count)+".jpg", right_eye_ball)
            # cv2.imwrite("../eyeImages/eye"+str(frame_count)+"_l.jpg", left_eye_ball)  

            # print(box_left)


            gaze, (x, y) = gaze_model.predict(left_eye, right_eye, orientation)

            # inferenceEnd = time.time()
            # inferenceTime = inferenceEnd - inferenceBegin
            # print("Inference Time of 4 models = " + str(inferenceTime))


            yaw = orientation[0]
            pitch = orientation[1]
            roll = orientation[2]

            sinY = math.sin(yaw * math.pi / 180.0)
            sinP = math.sin(pitch * math.pi / 180.0)
            sinR = math.sin(roll * math.pi / 180.0)

            cosY = math.cos(yaw * math.pi / 180.0)
            cosP = math.cos(pitch * math.pi / 180.0)
            cosR = math.cos(roll * math.pi / 180.0)

            cH, cW = crop_face.shape[:2]
            arrowLength = 0.5 * max(cH, cW)

            # Drawing Eye Boxes
            (p0_x, p0_y) = box_left[:2]
            (p12_x, p12_y) = box_left[2:4]
            cv2.rectangle(frame, (p0_x+xmin, p0_y+ymin), 
            						 (p12_x+xmin, p12_y+ymin-5), (255, 0, 0), 3)

            (p2_x, p2_y) = box_right[:2]
            (p17_x, p17_y) = box_right[2:4]
            cv2.rectangle(frame, (p2_x+xmin, p2_y+ymin), 
            						 (p17_x+xmin, p17_y+ymin-5), (255, 0, 0), 3)

            # to draw the eye points as circles 
            cv2.circle(frame, tuple(map(operator.add, p0, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p1, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p12, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p13, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p14, (xmin, ymin))), 1, (255, 0, 0), 2)

            # to draw the eye points as circles 
            cv2.circle(frame, tuple(map(operator.add, p2, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p3, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p15, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p16, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p17, (xmin, ymin))), 1, (255, 0, 0), 2)

            # to draw mouth points
            cv2.circle(frame, tuple(map(operator.add, p8, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p9, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p10, (xmin, ymin))), 1, (255, 0, 0), 2)
            cv2.circle(frame, tuple(map(operator.add, p11, (xmin, ymin))), 1, (255, 0, 0), 2)

            # Finding Eye Center
            xCenter_left = int((p0_x + p12_x) / 2) + xmin
            yCenter_left = int((p0_y + p12_y) / 2) + ymin
            leftEye_Center = (xCenter_left, yCenter_left)

            # Finding Eye Center
            xCenter_right = int((p2_x + p17_x) / 2) + xmin
            yCenter_right = int((p2_y + p17_y) / 2) + ymin
            rightEye_Center = (xCenter_right, yCenter_right)


            ############# DRAWING DIRECTION ARROWS BASED ON HEAD POSITION ############
            ## Euler angles to cartesian coordinates#
            # https://stackoverflow.com/questions/1568568/how-to-convert-euler-angles-to-directional-vector

            # Total rotation matrix is:

            # | cos(yaw)cos(pitch) -cos(yaw)sin(pitch)sin(roll)-sin(yaw)cos(roll) -cos(yaw)sin(pitch)cos(roll)+sin(yaw)sin(roll)|
            # | sin(yaw)cos(pitch) -sin(yaw)sin(pitch)sin(roll)+cos(yaw)cos(roll) -sin(yaw)sin(pitch)cos(roll)-cos(yaw)sin(roll)|
            # | sin(pitch)          cos(pitch)sin(roll)                            cos(pitch)sin(roll)|


            if visualizeHeadPose or controlMode == 2 or isCalibrated is False:

                # yaw and pitch are important for mouse control
                poseArrowX = orientation[0] #* arrowLength
                poseArrowY = orientation[1] #* arrowLength

                # Taking 2nd and 3rd row for 2D Projection
                ##############################LEFT EYE ###################################
                # cv2.arrowedLine(frame, leftEye_Center,
                #          (int((xCenter_left + arrowLength * (cosR * cosY - sinY * sinP * sinR))),
                #           int((yCenter_left + arrowLength * cosP * sinR))), (255, 0, 0), 2)
                
                # # center to top
                # cv2.arrowedLine(frame, leftEye_Center,
                #          (int(((xCenter_left - arrowLength * (sinY * sinP * cosR + cosY * sinR)))),
                #           int((yCenter_left + arrowLength * cosP * sinR))), (0, 0, 255), 2)

                # center to forward


                cv2.arrowedLine(frame, leftEye_Center, \
                         (int(((xCenter_left + arrowLength * sinY * cosP))), \
                          int((yCenter_left + arrowLength * sinP))), (0, 255, 0), 5)

                ##############################RIGHT EYE ###################################
                # cv2.arrowedLine(frame, rightEye_Center,
                #          (int((xCenter_right + arrowLength * (cosR * cosY - sinY * sinP * sinR))),
                #           int((yCenter_right + arrowLength * cosP * sinR))), (255, 0, 0), 2)
                
                # # center to top
                # cv2.arrowedLine(frame, rightEye_Center,
                #          (int(((xCenter_right - arrowLength * (sinY * sinP * cosR + cosY * sinR)))),
                #           int((yCenter_right + arrowLength * cosP * sinR))), (0, 0, 255), 2)

                # center to forward
                cv2.arrowedLine(frame, rightEye_Center,
                         (int(((xCenter_right + arrowLength * sinY * cosP))),
                          int((yCenter_right + arrowLength * sinP))), (0, 255, 0), 5)


            # gaze is required for calibration
            if visualizeGaze or controlMode == 1 or isCalibrated is False:

                gazeArrowX = gaze[0] * arrowLength
                gazeArrowY = -gaze[1] * arrowLength

                cv2.arrowedLine(frame, leftEye_Center,
                                (int(leftEye_Center[0] + gazeArrowX), 
                                 int(leftEye_Center[1] + gazeArrowY)), (0, 255, 0), 4)
                cv2.arrowedLine(frame, rightEye_Center,
                                (int(rightEye_Center[0] + gazeArrowX), 
                                 int(rightEye_Center[1] + gazeArrowY)), (0, 255, 0), 4)


            # print("Distance between mouth = " + str(p11[1] - p10[1]))

            ###############################
            # Compute Mouth Aspect Ratio  #
            ###############################
            mouthWidth = p9[0] - p8[0]
            mouthHeight = p11[1] - p10[1]

            if (mouthWidth != 0):
                mAspRatio = mouthHeight/ mouthWidth
            else:
                mAspRatio = 0
            # print('MAR RATIO = ' + str(mAspRatio))

            # mouthHeights.append(mouthHeight)
            # mouthMid = (np.max(mouthHeights) + np.min(mouthHeights)) / 2

            # if mouth is opened then trigger an event/ increment control mode.
            # if (len(mouthHeights) > 2 and 
            #     mouthHeight - mouthMid > 4):

            # To validate face is properly facing the camera.
            # To avoid erroneous control mode switches coz of face turns.
            if (isFaceInBounds(headYawPitchBounds, yaw, pitch) and mAspRatio > 0):
                # when mouth is open
                if mAspRatio > 0.4 and isMouthOpen is False:

                    print('increment control')
                    # mouthHeights.clear()
                    # isSoundControl = False
                    # Control will switch from gaze to head 
                    # to sound and then switch back to no control
                    if controlMode < 3:
                        controlMode += 1
                    else:
                        controlMode = 0
                    isMouthOpen = True

                elif mAspRatio > 0.3:
                    # Reset the click flag once smile is over.
                    isSmiling =  False
                elif mAspRatio < 0.3:
                    isMouthOpen = False
                # when mouth is wide, i.e. smiling    
                elif mAspRatio < 0.2 and isSmiling == False:
                    print('clicking  left')
                    isSmiling = True
                    mc.clickLeft()


            try:
                if frame_count % 5 == 0:

                    if (mc.calibrated is False):

                        isCalibrated = mc.captureCorners(gazeArrowX, gazeArrowY)

                        # print("xmin = " + str(mc.x_min))
                        # print("xmax = " + str(mc.x_max))
                        # print("ymin = " + str(mc.y_min))
                        # print("ymax = " + str(mc.y_max))
                        #ENABLE BELOW
                    else:
                        # Face should be forward facing inorder to take comamnds.
                        # if (isFaceInBounds(headYawPitchBounds, yaw, pitch)):

                        if controlMode == 1:

                            moveEnabled, lastGazes =  \
                                isMoveEnabled(lastGaze, stickinessGaze, gazeArrowX, gazeArrowY, lastGazes)

                            if moveEnabled:
                                print('moving mouse with gaze')
                                mc.moveWithGaze(gazeArrowX, gazeArrowY)
                                lastGaze = [gazeArrowX, gazeArrowY] #saving pos for stickiness
                        elif controlMode == 2:

                            moveEnabled, lastPoses =  \
                                isMoveEnabled(lastPose, stickinessHead, poseArrowX, poseArrowY, lastPoses)

                            if moveEnabled:
                                print('moving mouse with head. Yaw: ' + 
                                        str(poseArrowX) + " Pitch: " + str(poseArrowY) 
                                        + " Roll: " + str(orientation[2]))
                                mc.moveWithHead(poseArrowX, poseArrowY, headYawPitchBounds)
                                lastPose = [poseArrowX, poseArrowY] #saving pos for stickiness


            except Exception as err:
                print(traceback.format_exc())
                PrintException()
                logger.error("Exception occurred while moving cursor!")

            # Display calibration status on video
            if isCalibrated:
                frame = cv2.putText(frame, 'Calibration is done.', (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (0, 0, 255), 1, cv2.LINE_AA) 


            frame = cv2.putText(frame, 'Control Mode: ' + modes[controlMode], 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 0, 255), 1, cv2.LINE_AA)

            imshow('frame', frame, width=800)
            # frameEnd = time.time()
            # frameTime = frameEnd - frameBegin
            # print("FPS = " + str(1/frameTime))

    except Exception as err:
        print(traceback.format_exc())
        PrintException()
        logger.error(err)

    cv2.destroyAllWindows()
    feeder.close()

if __name__ == '__main__':

    # Can uncomment each line to get corresponding benchmark runs
	# To parse the video file given - all FP16 models
    # arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it video -d CPU -vh False -vg True -vf True'.split(' ')
    
    # To take input from the webcam - all FP16 models
    # arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True'.split(' ')

    # To take input from the webcam but with FP32 gaze & Landmark detection models
    # arg = '-f ../models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh False -vg True -vf True'.split(' ')

    # To take input from the webcam but with INT8 Face detection and FP32 gaze & Landmark detection models
    arg = '-f ../models/face-detection-adas-0001/FP32-INT8/face-detection-adas-0001.xml -l ../models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml -hp ../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -ge ../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i ../bin/demo.mp4 -it cam -d CPU -vh True -vg False -vf True'.split(' ')
    args = argparser().parse_args(arg)

    main(args)