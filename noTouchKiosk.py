
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
        stt.put(rh_result)



def load_device():
    """Reload audio device list"""
    device_list, default_input_index, loopback_index = \
                                audio_helper.get_input_device_list()

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

    return None, utters[0] #last word will return as reversed


def isFaceInBounds(yaw, pitch):

    minBound = -20
    maxBound = +20

    if yaw >= minBound and yaw <= maxBound and \
         pitch >= minBound and pitch <= maxBound:

        return True
    else:
        return False


def sigmoid(x, L ,x0, k, b):

    y = L / (1 + np.exp(k*(x-x0)))+b
    return (y)


def isCurveSigmoid(pixelCounts, count):

    try:
        xIndex = len(pixelCounts)

        p0 = [max(pixelCounts), np.median(xIndex),1,min(pixelCounts)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, list(range(xIndex)), pixelCounts, p0, method='lm', maxfev=5000)

        yVals = sigmoid(list(range(xIndex)), *popt)

        # May have to check for a value much less than Median to avoid false positives.
        if np.median(yVals[:10]) - np.median(yVals[-10:]) > 15:
            print('Triggered Event')
            return True

    except Exception as err:
        print(traceback.format_exc())

    return False




def findCurveFit(eye, image, pixelCount, frame_count, numFrames = 50):

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




def findClosurebyStats(eye, image, pixelCount, frame_count, samples = 20, numFrames = 50):
    
    triggerEvent = False
    medianDiff = 0

    if (len(image) == 0):
        return pixelCount, False, 0

    # Convert to gray scale as histogram works well on 256 values.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate frequency of pixels in range 0-255 
    histg = cv2.calcHist([gray],[0],None,[256],[0,256])

    # hack to know whether eye is closed or not.
    # more spread of pixels in a histogram signifies an opened eye
    activePixels = np.count_nonzero(histg)
    pixelCount.append(activePixels)

    if len(pixelCount) > numFrames:

        tailCounts = pixelCount[-numFrames+10:]

        begin = tailCounts[:samples]

        end = tailCounts[-samples:]

        medianDiff = np.median(begin) - np.median(end)

        if  medianDiff > 10 and np.std(begin) < 5 and np.std(end) < 5:
            print('Event Triggered')
            pixelCount.clear()
            triggerEvent = True

    return pixelCount, triggerEvent, medianDiff


def findPeaks(image, pixelCount, frame_count, numFrames = 50):

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

    if len(pixelCount) > numFrames:

        diff = np.diff(pixelCount[-numFrames+10:])

        peaks = peakutils.peak.indexes(np.array(diff), thres=0.8, min_dist=2)
        x =   np.array([i * -1 for i in diff])
        peaksReflected = peakutils.peak.indexes(np.array(x), thres=0.8, min_dist=2)

        # if peak is there on upright and reflected signal then the closed eyes are open soon
        # i.e. it denotes a blink and not a gesture. But if peak is found only on the reflected
        # signal then eyes are closed for long time to indicate gesture.
        if (peaksReflected.size > 0 and x[peaksReflected[0]] > 0 and peaks.size == 0):
            print('Event Triggered...')

            pixelCount.clear()
            triggerEvent = True

    return pixelCount, triggerEvent



def hikeControlMode(controlMode):

    print('increment control')
    # Control will switch from gaze to head 
    # to sound and then switch back to no control
    if controlMode < 3:
        controlMode += 1
    else:
        controlMode = 0

    return controlMode

def dipControlMode(controlMode):

    print('decrement control')
    # Control will switch from gaze to head 
    # to sound and then switch back to no control
    if controlMode > 0:
        controlMode -= 1
    else:
        controlMode = 3

    return controlMode


def imshow(windowname, frame, width=None):

    if width is not None:
        frame = imutils.resize(frame, width=width)

    # cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, frame)
    # cv2.resizeWindow(windowname, 300,300)
    cv2.waitKey(25)

def isMoveSignificant(lastPosition, stickiness, x, y):

    last_x = lastPosition[0]
    last_y = lastPosition[1]

    # print("Previous X = " + str(last_x) + ". Previous Y = " + \
    #     str(last_y) + ". Current X = " + str(x) + ". Current Y = " + str(y))

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
    # Left Click = Yawn 
    # Right Click = Looking up
    # Increment Control Modes = Right Wink
    # Left Eye Wink and Smile are left unassigned
    # You can dictate text in Sound mode (Control mode = 3)


    #####################################################################  
    # Initializing the Speech Recognition Thread
    #####################################################################  

    # You can add more controls as you deem fit.
    numbers = ['zero', 'one', 'two', 'three', 'four', \
                'five', 'six', 'seven', 'eight', 'nine']

    controls = ['left', 'right', 'up', 'down']

    control_syn = {}
    for control in controls:
        control_syn.setdefault(control, [])

    # Need to account for similar sounding words as speech recog is on the edge!
    control_syn['left'].extend(['let', 'left', 'light', 'live', 'laugh'])
    control_syn['right'].extend(['right', 'write', 'great', 'fight', 'might', 'ride'])
    control_syn['up'].extend(['up', 'hop', 'hope', 'out'])
    control_syn['down'].extend(['down', 'doubt', 'though'])

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

    # Fixing 60x60 box as yaw and pitch boundaries to
    # correspond to head turning left and right (yaw)
    # and also moving up and down (pitch)
    headYawPitchBounds = [-30, 30]

    lastGaze = [0, 0]
    lastPose = [0, 0]

    # Set the stickiness value
    stickinessHead = 5
    stickinessGaze = 10

    eventText = "No Event"

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
    islookingUp = False

    lastPoses = collections.deque(maxlen=20)
    lastGazes = collections.deque(maxlen=20)

    try:
        frame_count = 0

        for ret, frame in feeder.next_batch():

            ################################################################
            # if any sound is deciphered from the spunned off thread then 
            # check the last 3 words of the utterance for matching control word
            if (stt.qsize() > 0 and controlMode == 3):
                
                utterance = stt.get()
                print("From Parent: " + utterance)
                
                # need to process again only if change in utterance
                if (prevUtterance != utterance):
                    control, lastWord = detectSoundEvent(utterance, controls, control_syn)

                    if control is not None:

                        direction = controls.index(control)
                        mc.moveRelative(direction)

                    else:

                        if lastWord in numbers:
                            lastWord = str(numbers.index(lastWord))

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

            # Draw the face box
            xmin, ymin, xmax, ymax = box
            if visualizeFace:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 3)

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


            pad = 10
            # Compute Right Eye: Close Snap
            right_eye_ball = frame[ymin + p1[1] - pad: ymin + p0[1] + pad,
                                    xmin + p1[0] - pad: xmin + p0[0] + pad]

            # Compute Left Eye: Close Snap
            left_eye_ball = frame[ymin + p3[1] - pad: ymin + p2[1] + pad,
                                    xmin + p2[0] - pad: xmin + p3[0] + pad]


            # pixelCount_leye_bk = pixelCount_leye #can delete this line
            pixelCount_reye, Rtrigger, probR = findClosurebyStats('Right', right_eye_ball, pixelCount_reye, frame_count) 
            pixelCount_leye, Ltrigger, probL = findClosurebyStats('Left', left_eye_ball, pixelCount_leye, frame_count) 


            print("probL: " + str(probL))
            if probL < -30 and islookingUp is False:
                print('Click Right')
                controlMode = hikeControlMode(controlMode) ## to change
                # mc.clickRight()
                islookingUp = True
                eventText = 'Increment Control Mode'
            elif probL > 0:
                islookingUp = False
                if (eventText == 'Increment Control Mode'):
                    eventText = 'No Event'


            # If both eyes are detected as pressed (as one eye
            # can shrink when the other eye is winked) then check 
            # which eye has higher probability of closure.
            # Note: To close both eyes is not a gesture.
            if Ltrigger and Rtrigger:
                # print("probR = " + str(probR) + "probL = " + str(probL))
                if probR > probL:
                    Ltrigger = False
                else:
                    Rtrigger = False

            # If you want to enable left and right wink actions, 
            # then call corresponding functions here.
            if Ltrigger:
                print('left eye pressed')
                # controlMode = dipControlMode(controlMode)
                # writeList(pixelCount_leye_bk) # Dumping list for debugging purpose
                # mc.scroll(20) # you can pass the head pose up/down as param
                # mc.drag()
            
            if Rtrigger:
                print('right eye pressed')
                # controlMode = hikeControlMode(controlMode)
                # mc.clickRight()
            

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

            # Total rotation matrix is: (See correct matrix in blog)

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
                #          (int((xCenter_left + arrowLength * (cosR * cosY + sinY * sinP * sinR))),
                #           int((yCenter_left + arrowLength * cosP * sinR))), (255, 0, 0), 4)
                
                # # center to top
                # cv2.arrowedLine(frame, leftEye_Center,
                #          (int(((xCenter_left + arrowLength * (sinY * sinP * cosR - cosY * sinR)))),
                #           int((yCenter_left + arrowLength * cosP * cosR))), (0, 0, 255), 4)

                # center to forward
                # cv2.arrowedLine(frame, leftEye_Center, \
                #          (int(((xCenter_left + arrowLength * sinY * cosP))), \
                #           int((yCenter_left - arrowLength * sinP))), (0, 255, 0), 4)

                ##############################RIGHT EYE ###################################
                # cv2.arrowedLine(frame, rightEye_Center,
                #          (int((xCenter_right + arrowLength * (cosR * cosY + sinY * sinP * sinR))),
                #           int((yCenter_right + arrowLength * cosP * sinR))), (255, 0, 0), 4)
                
                # # center to top
                # cv2.arrowedLine(frame, rightEye_Center,
                #          (int(((xCenter_right + arrowLength * (sinY * sinP * cosR - cosY * sinR)))),
                #           int((yCenter_right + arrowLength * cosP * cosR))), (0, 0, 255), 4)

                # center to forward
                # cv2.arrowedLine(frame, rightEye_Center,
                #          (int(((xCenter_right + arrowLength * sinY * cosP))),
                #           int((yCenter_right - arrowLength * sinP))), (0, 255, 0), 4)


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

            # To validate face is properly facing the camera.
            # To avoid erroneous control mode switches coz of face turns.
            if (isFaceInBounds(yaw, pitch) and mAspRatio > 0):

                # These threshold constants need to either modified or made dynamic.
                # 
                # when mouth is open
                if mAspRatio > 0.4 and isMouthOpen is False:

                    # mouthHeights.clear()
                    # isSoundControl = False
                    print('clicking left')
                    mc.clickLeft()
                    isMouthOpen = True
                    eventText = 'Click Left'

                elif mAspRatio < 0.35:
                    isMouthOpen = False
                    if (eventText == 'Click Left'):
                        eventText = 'No Event'

                # when mouth is wide, i.e. smiling    
                if mAspRatio < 0.26 and isSmiling == False:

                    print('You are smiling...')
                    eventText = 'Smiling'
                    isSmiling = True
                    
                elif mAspRatio > 0.3:
                    # Reset the click flag once smile is over.
                    isSmiling =  False
                    if (eventText == 'Smiling'):
                        eventText = 'No Event'

            # controlMode = 3 # To debug a specific control mode.


            try:
                if frame_count % 5 == 0:

                    if (mc.calibrated is False):

                        isCalibrated = mc.captureCorners(gazeArrowX, gazeArrowY)

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

            frame = cv2.putText(frame, 'Event: ' + eventText, 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 1, cv2.LINE_AA)

            frame = cv2.putText(frame, 'MAR: ' + str(round(mAspRatio, 2)), 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 1, cv2.LINE_AA)

            frame = cv2.putText(frame, 'Mouse Loc: ' + str(mc.getLocation()), 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 1, cv2.LINE_AA)
            

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