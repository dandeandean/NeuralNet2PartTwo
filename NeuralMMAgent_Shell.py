# Some potentially useful modules
import random
import numpy
import math
import matplotlib.pyplot as plt
import time
import copy


class NeuralMMAgent(object):
    '''
    Class to for Neural Net Agents that compete in the Mob task
    '''

    def __init__(self, num_in_nodes, num_hid_nodes, num_hid_layers, num_out_nodes, \
                learning_rate = 0.2, max_epoch=10000, min_sse=.01, momentum=0, \
                creation_function=None, activation_function=None, random_seed=1):
        '''
        Arguments:
            num_in_nodes -- total # of input nodes for Neural Net
            num_hid_nodes -- total # of hidden nodes for each hidden layer
                in the Neural Net
            num_hid_layers -- total # of hidden layers for Neural Net
            num_out_nodes -- total # of output layers for Neural Net
            learning_rate -- learning rate to be used when propogating error
            max_epoch -- maximum number of epochs for our NN to run during learning
            min_sse -- minimum SSE that we will use as a stopping point
            momentum -- Momentum term used for learning
            creation_function -- function that will be used to create the
                neural network given the input
            activation_function -- list of two functions:
                1st function will be used by network to determine activation given a weighted summed input
                2nd function will be the derivative of the 1st function
            random_seed -- used to seed object random attribute.
                This ensures that we can reproduce results if wanted
        '''
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and\
            num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"


    def train_net_incremental(self, input_list, output_list, max_num_epoch=100000, \
                    min_sse=0.001):
        ''' Trains neural net using incremental learning
            (update once per input-output pair)
            Arguments:
                input_list -- 2D list of inputs
                output_list -- 2D list of outputs matching inputs
            Outputs:
                1d list of errors (total error each epoch) (e.g., [0.1])
        '''
                #Some code...#
            #all_err.append(total_err)

            #if (total_err < min_sse):
                #break

    def _feed_forward(self, input_list, row):
        '''Used to feedforward input and calculate all activation values
            Arguments:
                input_list -- a list of possible input values
                row -- the row from that input_list that we should use
            Outputs:
                list of activation values
        '''
        pass

    def _calculate_deltas(self, activations, errors, prev_weight_deltas=None):
        '''Used to calculate all weight deltas for our neural net
            Parameters:
                activations -- a 2d list of activation values
                errors -- a 2d list of errors
                prev_weight_deltas [OPTIONAL] -- a 2d list of previous weight deltas
            Output:
                A tuple made up of 3 items:
                    A 2d list of little deltas (e.g., [[0, 0], [-0.1, 0.1], [0.1]])
                    A 2d list of weight deltas (e.g., [[-0.1, 0.1, -0.1, 0.1], [0.1, 0.1]])
                    A 2d list of bias deltas (e.g., [[0, 0], [-0.1, 0.1], [0]])
        '''

        #Calculate error gradient for each output node & propgate error
        #   (calculate weight deltas going backward from output_nodes)




    def _adjust_weights_bias(self, weight_deltas, bias_deltas):
        '''Used to apply deltas
        Parameters:
            weight_deltas -- 2d list of weight deltas
            bias_deltas -- 2d list of bias deltas
        Outputs:
            A tuple w/ the following items (in order):
            2d list of all weights after updating (e.g. [[-0.071, 0.078, 0.313, 0.323], [-0.34, 0.021]])
            list of all biases after updating (e.g., [[0, 0], [0, 0], [0]])
        '''

    #########ACCESSORS

    def get_weights(self):
        return (self.weights)

    def set_weights(self, weights):
        self.weights = weights

    def get_bias(self):
        return (self.bias)

    def set_biases(self, bias):
        self.bias = bias

    ################

    @staticmethod
    def sigmoid_af(summed_input):
        #Sigmoid function
        pass

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        #the derivative of the sigmoid function
        pass


#----#
#Some quick test code
"""
test_agent = NeuralMMAgent(2, 2, 1, 1,random_seed=5, max_epoch=1000000, \
                            learning_rate=0.2, momentum=0)
test_in = [[1,0],[0,0],[1,1],[0,1]]
test_out = [[1],[0],[0],[1]]
test_agent.set_weights([[-.37,.26,.1,-.24],[-.01,-.05]])
test_agent.set_biases([[0,0],[0,0],[0,0]])
all_errors = test_agent.train_net(test_in, test_out, max_sse = test_agent.max_sse, \
                     max_num_epoch = test_agent.max_epoch)
"""
