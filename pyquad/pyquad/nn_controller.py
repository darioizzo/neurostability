import pickle
import numpy as np
from pyaudi import gdual_double, tanh, exp, log

class Controller:
    def __init__(self, path_to_pickle):
        self.input_scaler_params, self.output_scaler_params, self.config, self.weights \
            = pickle.load(open(path_to_pickle, 'rb'))

    def preprocess_input(self, state):
        return (state - self.input_scaler_params[0]) / self.input_scaler_params[1]

    def postprocess_output(self, pred):
        out = (pred - self.output_scaler_params[0]) / self.output_scaler_params[1]
        return (out * self.output_scaler_params[2]) + self.output_scaler_params[3]

    def nn_predict(self, model_input):
        vector = model_input
        dense_layer_count = 0
        for layer_config in self.config:
            if layer_config['class_name'] == 'Dense':
                wgts, biases = self.weights[dense_layer_count*2 : (dense_layer_count+1)*2]
                vector = wgts.T.dot(vector) + biases
                dense_layer_count += 1
            elif layer_config['class_name'] == 'Activation':
                if layer_config['config']['activation'] == 'relu':
                    vector[convert_gdual_to_float(vector) < 0] = 0
                elif layer_config['config']['activation'] == 'tanh':
                    vector = np.vectorize(lambda x : tanh(x))(vector)
                elif layer_config['config']['activation'] == 'softplus':
                    vector = np.vectorize(lambda x : exp(x))(vector) + 1
                    vector = np.vectorize(lambda x : log(x))(vector)
        return vector


    def compute_control(self, state):
        model_input = self.preprocess_input(state)
        model_pred = self.nn_predict(model_input)
        control_out = self.postprocess_output(model_pred)
        return control_out
    
    def convert_gdual_to_float(gdual_array):
        floatize = lambda x : x.constant_cf if type(x) == gdual_double else x
        convert_to_float = np.vectorize(floatize, otypes=[np.float64])
        return convert_to_float(gdual_array)
