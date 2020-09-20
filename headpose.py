'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import cv2
import numpy as np

import logging

class HeadPoseDetect:
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
        return self.preprocess_output(self.exec_network.requests[0].outputs)


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
        
        if len(image) == 0:
            return image

        net_input_shape = self.network.inputs[self.input].shape
        frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        frame = frame.transpose(2, 0, 1)
        frame = frame.reshape(1, *frame.shape)
        return frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        return np.array([outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]])
