'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import cv2
import numpy as np

class FaceDetector:
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

        cropface, box = self.preprocess_output(result[0][0], image)
        return cropface, box


    def check_model(self):
        supported_layers = self.core.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("These are unsupported layers: " + str(unsupported_layers))
            exit(1)
        print("All Face Detection Model layers are supported in this device")

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

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        cords = []
        for id, label, confidence, x_min, y_min, x_max, y_max in outputs:
            if confidence > .7:
                width = x_max - x_min
                height = y_max - y_min
                cords.append([x_min, y_min, x_max, y_max])

        box = cords[0]
        h, w = image.shape[0:2]
        box = box * np.array([w, h, w, h])
        box = box.astype(np.int32)
        x_min, y_min, x_max, y_max = box
        cropped_face = image[y_min:y_max, x_min:x_max]
        return cropped_face, box
