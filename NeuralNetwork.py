import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layers, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        
        self.layers = 2 + hidden_layers
        self.generate_weights()
        
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        outputs = [inputs]
        
        for i in range(self.layers - 1):
            outputs.append(self.activation(np.dot(self.weights[i], outputs[i])))
        
        errors = [targets - outputs[-1]]
        
        reversed_weights = self.weights[::-1]
        
        for i in range(self.layers - 2):
            errors.append(np.dot(reversed_weights[i].T, errors[i]))
        
        errors = errors[::-1]
        
        for i in reversed(range(self.layers - 1)):
            self.weights[i] += self.lr * np.dot(errors[i] * outputs[i+1] * (1 - outputs[i+1]), outputs[i].T)
        
        return outputs[-1]
    
    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        
        outputs = [inputs]
        
        for i in range(self.layers - 1):
            outputs.append(self.activation(np.dot(self.weights[i], outputs[i])))
        
        return outputs[-1]
        
    def activation(self, x):
        return scipy.special.expit(x)
    
    def generate_weights(self):
        self.weights = [np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5]
        
        for x in range(self.layers - 3):
            self.weights.append(np.random.rand(self.hidden_nodes, self.hidden_nodes) - 0.5)
        
        self.weights.append(np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)