'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import math
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import cv2
import numpy as np

import logging

class GazeDetect:
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

        # self.network = self.core.read_network(model=str(model_name),
        #                                       weights=str(os.path.splitext(model_name)[0] + ".bin"))
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

    def predict(self, left_eye, right_eye, orientation):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)

        # processed_frame = self.preprocess_input(image)
        # inference_start_time = time.time()
        self.exec_network.start_async(request_id=self.request_id,
                                      inputs={'left_eye_image': processed_left_eye,
                                              'right_eye_image': processed_right_eye,
                                              'head_pose_angles': orientation})
        # self.exec_network.requests[self.request_id].wait()

        if self.exec_network.requests[self.request_id].wait(-1) == 0:
            result = self.exec_network.requests[self.request_id].outputs[self.output]
            # self.exec_network.requests[0].wait()
            cords = self.preprocess_output(result[0], orientation)

            # print(cords)
            return result[0], cords
            

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
        net_input_shape = self.network.inputs['right_eye_image'].shape

        frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        frame = frame.transpose(2, 0, 1)
        frame = frame.reshape(1, *frame.shape)
        return frame

    def preprocess_output(self, output, head_position):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        roll = head_position[2]
        gaze_vector = output / cv2.norm(output)
        # print('roll = ')
        # print(roll)
        # print('gaze vector = ')
        # print(gaze_vector)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)

        x = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue + gaze_vector[1] * cosValue
        return (-x*10, y*10)