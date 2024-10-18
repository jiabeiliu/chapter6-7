class NeuralNetwork:
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.activation_functions = activation_functions
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        for l in range(1, len(self.layers)):
            parameters[f"W{l}"] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
            parameters[f"b{l}"] = np.zeros((self.layers[l], 1))
        return parameters

    def forward_propagation(self, X):
        A = X
        cache = {}
        for l in range(1, len(self.layers)):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(W, A) + b
            
            if self.activation_functions[l-1] == 'linear':
                A = Activation.linear(Z)
            elif self.activation_functions[l-1] == 'relu':
                A = Activation.relu(Z)
            elif self.activation_functions[l-1] == 'sigmoid':
                A = Activation.sigmoid(Z)
            elif self.activation_functions[l-1] == 'tanh':
                A = Activation.tanh(Z)
            elif self.activation_functions[l-1] == 'softmax':
                A = Activation.softmax(Z)
            
            cache[f"A{l}"] = A
            cache[f"Z{l}"] = Z
        
        return A, cache

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(AL + 1e-15))  # Add epsilon to avoid log(0)
        return cost
