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
        assert num_in_nodes > 0 and num_hid_layers > 0 and num_hid_nodes and num_out_nodes > 0, "Illegal number of input, hidden, or output layers!"
        self.num_in_nodes = num_in_nodes
        self.num_hid_nodes = num_hid_nodes
        self.num_hid_layers = num_hid_layers
        self.num_out_nodes = num_out_nodes
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.min_sse = min_sse
        self.momentum = momentum
        self.creation_function = creation_function
        self.activation_function = activation_function
        self.random_seed = random_seed

        #WE CAN SIMPLY FILL OUT THESE VALUES LATER AS WE CALCULATE THEM, BUT WE INIT HERE
        self.weights = self._construct_ligaments()
        self.weight_ds = self._construct_ligaments()
        
        self.activations = self._construct_skeleton()
        self.errors = self._construct_skeleton()
        self.biases = self._construct_skeleton()
        self.bias_ds = self._construct_skeleton()
                
    ##################################################################################################
    #ACCESSORS
    def get_weights(self):
        return (self.weights)

    def set_weights(self, weights):
        self.weights = weights

    def get_biases(self):
        return (self.biases)

    def set_biases(self, bias):
        self.bias = biases
    
    def set_thetas(self, thetas):
        self.thetas = thetas
    @staticmethod
    def sigmoid_af(summed_input):
        #Sigmoid function
        e_to_the = numpy.exp #pythonic
        denom = 1+e_to_the(-summed_input)
        return 1/denom 

    @staticmethod
    def sigmoid_af_deriv(sig_output):
        #the derivative of the sigmoid function"
        return sig_output * (1-sig_output)

    def matrixify_weigths(self,layer):
        layer_in_question = self.weights[layer]
        cols = list()
        for i in range(len(layer_in_question)//self.num_hid_nodes):
            cols.append(self._get_incoming_weights(layer+1,i))
        return cols
    
        
    def _construct_skeleton(self):
        #returns a 2d array that represents each node, inits to all zeros
        #the middle layers
        out=numpy.zeros((self.num_hid_layers,self.num_hid_nodes)).tolist()
        #the first layer 
        out.insert(0,[0.0] * self.num_in_nodes)
        #the last layer
        out.append([0.0] * self.num_out_nodes)
        return out
    ##################################################################################################
    #FEED FORWARD
    def _feed_forward(self, input_list, row):
        '''Used to feedforward input and calculate all activation values
            Arguments:
                input_list -- a list of possible input values
                row -- the row from that input_list that we should use
            Outputs:
                list of activation values
        '''
        return self.__feedforward(input_list[row])
    
    
    
    
    def __feedforward(self,inp):
        #simply shoe the input layer in there
        self.activations = [ [float(i) for i in inp] ]
        # with biases
        for i in range(len(self.activations[0])): self.activations[0][i] += self.bias[0][i]
        for i in range(self.num_hid_layers+1):
            # change the order of the weights so we can take the dot product with the previous layer of activations 
            self.activations.append(NeuralMMAgent.sigmoid_af(numpy.dot(numpy.matrix(self.matrixify_weigths(i)),self.activations[-1])+self.bias[i+1]).tolist()[0])
        return self.activations
    
    def _get_incoming_weights(self,layer_index,node_index):
        incoming_weights = list()
        if not layer_index: return [] #zeroth layer is input should not have incoming weights
                                      # is it a hidden node?                        or is it an output node?
        depth = self.num_hid_nodes*(layer_index <= self.num_hid_layers) + self.num_out_nodes*(layer_index > self.num_hid_layers)
        for i in range(node_index,len(self.weights[layer_index-1]),depth):
            incoming_weights.append(self.weights[layer_index-1][i])
        return incoming_weights

    
    def _get_outgoing_weights(self,layer_index,node_index):
        if layer_index == self.num_hid_layers+1: return []
        assert(len(self.weights[layer_index])>(node_index))
        
        outgoing_weights = list()
        depth = self.num_hid_nodes*(layer_index <= self.num_hid_layers) + self.num_out_nodes*(layer_index > self.num_hid_layers)
        if layer_index == self.num_hid_layers+1: return []
        #a straightforward way of getting the # of nodes in the next layer 
        nodes_in_next_layer = self.num_hid_nodes*(layer_index+1 <= self.num_hid_layers) + self.num_out_nodes*(layer_index+1 > self.num_hid_layers)
        for i in range(0,nodes_in_next_layer):
            outgoing_weights.append(self.weights[layer_index][node_index*nodes_in_next_layer +i])
        return outgoing_weights
    ##################################################################################################
    
    
    def _construct_ligaments(self):
        #returns a 2d array that represents the connections, inits to all zeros
        #hidden nodes connections
        out = numpy.zeros((self.num_hid_layers-1,self.num_hid_nodes**2)).tolist()
        #input nodes to hidden nodes
        out.insert(0,[0.0]*self.num_in_nodes*self.num_hid_nodes)
        #last layer to output nodes connections
        out.append([0.0]*self.num_hid_nodes*self.num_out_nodes)
        return out
