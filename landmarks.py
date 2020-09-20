'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import cv2
import numpy as np

import logging

class LandmarksDetect:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.mode = 'async'
        self.exec_network = None
        self.device = device
        self.request_id = 0

        self.net = IENetwork(model=str(model_name),
                        weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.plugin = IEPlugin(device=device)

        self.core = IECore()
        self.network = self.plugin.load(network=self.net, num_requests=2)

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

        self.logger = logging.getLogger()

        self.check_model()        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.net, self.device)
        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_frame = self.preprocess_input(image)

        self.exec_network.start_async(request_id=self.request_id,
                                      inputs={self.input: processed_frame})

        self.exec_network.requests[self.request_id].wait()
        result = self.exec_network.requests[self.request_id].outputs[self.output]

        self.exec_network.requests[0].wait()
        return self.preprocess_output(self.exec_network.requests[0].outputs, image)


    def check_model(self):
        supported_layers = self.core.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            self.logger.error("These are unsupported layers: " + str(unsupported_layers))
            raise Exception('There are unsupported layers!')
        self.logger.info("All Face Detection Model layers are supported in this device")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.network.inputs[self.input].shape
        frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        frame = frame.transpose(2, 0, 1)
        frame = frame.reshape(1, *frame.shape)
        return frame

    def scaleBoxes(self, p, w, h):

        return tuple((p * np.array([w, h])).astype(np.int32))

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # https://docs.openvinotoolkit.org/latest/omz_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html

        h, w = image.shape[0:2]
        paddingConstant = 10

        landmarks = outputs['align_fc3']

        # Computing the left eye box corners
        box = (landmarks[0][0], landmarks[0][1], landmarks[0][12*2], landmarks[0][12*2+1])
        box_left = box * np.array([w, h, w, h])
        box_left = box_left.astype(np.int32) + \
            [paddingConstant, paddingConstant, -paddingConstant, -paddingConstant]

        left_eye = image[box_left[3]:box_left[1], box_left[2]:box_left[0]]

        # Computing the right eye box corners
        box = (landmarks[0][2*2], landmarks[0][2*2+1], landmarks[0][2*17], landmarks[0][2*17+1])
        box_right = box * np.array([w, h, w, h])    
        box_right = box_right.astype(np.int32) + \
            [-paddingConstant, paddingConstant, paddingConstant, -paddingConstant]

        right_eye = image[box_right[3]:box_right[1], box_right[0]:box_right[2]]

        # Computing the left eye landmarks
        p0 = self.scaleBoxes((landmarks[0][0], landmarks[0][1]), w, h)
        p1 = self.scaleBoxes((landmarks[0][1*2], landmarks[0][1*2+1]), w, h)
        p12 = self.scaleBoxes((landmarks[0][12*2], landmarks[0][12*2+1]), w, h)
        p13 = self.scaleBoxes((landmarks[0][13*2], landmarks[0][13*2+1]), w, h)
        p14 = self.scaleBoxes((landmarks[0][14*2], landmarks[0][14*2+1]), w, h)

        # Computing the right eye landmarks
        p2 = self.scaleBoxes((landmarks[0][2*2], landmarks[0][2*2+1]), w, h)
        p3 = self.scaleBoxes((landmarks[0][3*2], landmarks[0][3*2+1]), w, h)
        p15 = self.scaleBoxes((landmarks[0][15*2], landmarks[0][15*2+1]), w, h)
        p16 = self.scaleBoxes((landmarks[0][16*2], landmarks[0][16*2+1]), w, h)
        p17 = self.scaleBoxes((landmarks[0][17*2], landmarks[0][17*2+1]), w, h)       

        # Computing the mouth landmarks
        p8 = self.scaleBoxes((landmarks[0][8*2], landmarks[0][8*2+1]), w, h)
        p9 = self.scaleBoxes((landmarks[0][9*2], landmarks[0][9*2+1]), w, h) 
        p10 = self.scaleBoxes((landmarks[0][10*2], landmarks[0][10*2+1]), w, h)
        p11 = self.scaleBoxes((landmarks[0][11*2], landmarks[0][11*2+1]), w, h)    

        return box_left, box_right, left_eye, right_eye, \
                p0, p1, p12, p13, p14, p2, p3, p15, p16, p17, p8, p9, p10, p11