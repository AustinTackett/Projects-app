import numpy as np
import sklearn

class SimpleNeuralNetwork(object):
    def __init__(self, num_inputs=4, hidden_layers=[3,3], num_outputs=2, activation='Tanh'):
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        np.random.seed(0);
        
        layers = [num_inputs] + hidden_layers + [num_outputs]
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1) ]
        self.hidden_layer = [np.zeros((layers[i], layers[i+1])) for i in range(len(layers) - 1) ] 
        self.activations = [np.zeros(layers[i]) for i in range(len(layers))]
        self.derivatives = [np.zeros((layers[i], layers[i + 1])) for i in range(len(layers) - 1)]
        
        match activation :
            case 'Tanh':
                self._activation_function = self._tanh
                self._activation_function_derivative = self._tanh_derivative
            case 'Sigmoid':
                self._activation_function = self._sigmoid
                self._activation_function_derivative = self._sigmoid_derivative
            case _:
                self._activation_function = self._tanh
                self._activation_function_derivative = self._tanh_derivative

    def forward_propagate(self, inputs):
        InitialInput = inputs
        self.activations[0] = InitialInput

        for i, weightMatrix in enumerate(self.weights):
            self.hidden_layer[i] = np.dot(self.activations[i], weightMatrix)
            self.activations[i+1] = self._activation_function(self.hidden_layer[i])

        return self.activations[-1]
    
    def back_propagate(self, output, target):
        dCost_dActivation = np.array(target - output)
    
        for i in reversed(range(len(self.derivatives))):
            CurrentLayerActivation = np.expand_dims(self.activations[i + 1], axis=0)
            PrevLayerActivation = np.expand_dims(self.activations[i], axis=0)
            
            '''(DEPRECATED B/C SLOW) USING MATRIX MULTIPLICATION USING JACOBIAN REPRESENTATION LEFT FOR REFERENCE
            dActivation_dHiddenLayer = np.diag(self._tanh_derivative(CurrentLayerActivation))
            deltaAtCurrentLayer = np.dot(dCost_dActivation, dActivation_dHiddenLayer)'''
            
            dActivation_dHiddenLayer = self._activation_function_derivative(CurrentLayerActivation)
            deltaAtCurrentLayer = np.multiply(dCost_dActivation, dActivation_dHiddenLayer)
            
            dHiddenLayer_dWeights = PrevLayerActivation
            '''Each column is the scalar (cost wrt the hidden layer n) multiplied by the vector representation of the prev layers activation
            Because each hidden layer wrt the weights IS THE SAME previous activation layer, we can use an outer product like operation to multiply each delta by the 
            by the previous activation layer to map two vectors onto a matrix of the correct dimensions and values.'''
            self.derivatives[i] = np.dot(dHiddenLayer_dWeights.T, deltaAtCurrentLayer)
            
            '''Calculate wrt to activation instead of weight to update delta for previous layer. 
            The transposed weight matrix is equivalent to the jacobian of the hidden layers wrt each activation node'''
            dCost_dActivation = np.dot(deltaAtCurrentLayer, self.weights[i].T)
    
    def gradient_descent(self, learning_rate, derivatives):
        for weightMatrix, derivativeMatrix in zip(self.weights, derivatives):
            weightMatrix += learning_rate * derivativeMatrix
            
                
    def train(self, data, target_label, batch_size, epochs, learning_rate, shuffle=True):
        for _ in range(epochs):
            if(shuffle):
                data, target_label = sklearn.utils.shuffle(data, target_label)
            epochSumCost = 0
            
            #Iterate through data with spliced batches
            for i in range(0, len(data), batch_size):
                '''The expected behavior is for the data to already be formatted in a list of one dimensional ND arrays with a shape of [num_inputs,].
                The data input is then sliced into batches of specified size'''
                sample_batch = data[i: i + batch_size]
                target_batch = target_label[i: i + batch_size]

                #Iterate through each batch
                batch_derivative_accumulator = [np.zeros_like(deriv_layer) for deriv_layer in self.derivatives]
                for sample, target in zip(sample_batch, target_batch):
                    prediction = self.forward_propagate(sample)
                    self.back_propagate(prediction, target)  
                    self.gradient_descent(learning_rate, self.derivatives)
                    batch_derivative_accumulator = [np.add(acc_layer, deriv_layer) for acc_layer, deriv_layer in zip(batch_derivative_accumulator, self.derivatives)]
                    epochSumCost += (target - prediction)**2
                
                mean_batch_derivatives = [layer_accumulated * (1/batch_size) for layer_accumulated in batch_derivative_accumulator]
                self.gradient_descent(learning_rate, mean_batch_derivatives)
        
    def predict(self, input):
        activation = input

        for weightMatrix in self.weights:
            hiddenLayerOutput = np.dot(activation, weightMatrix)
            activation = self._activation_function(hiddenLayerOutput)


        return activation
    
    def _sigmoid(self, x):
        return np.array(1/(1 + np.exp(-x)))

    def _sigmoid_derivative(self, x):
        return np.array(x * (1.0 - x))
    
    def _tanh(self, x):
        return np.array((2/(1 + np.exp(-2 * x))) - 1)

    def _tanh_derivative(self, x):
        return np.array(1 - (x ** 2))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0) * 1.0
        
        