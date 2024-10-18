    def backward_propagation(self, X, Y, cache):
        m = Y.shape[1]
        L = len(self.layers) - 1  # Number of layers excluding input
        grads = {}
        
        # Compute the derivative of the cost with respect to AL
        AL = cache[f"A{L}"]
        dAL = - (np.divide(Y, AL + 1e-15) - np.divide(1 - Y, 1 - AL + 1e-15))  # Gradient for softmax + cross-entropy

        for l in reversed(range(1, L + 1)):
            A_prev = cache[f"A{l-1}"] if l > 1 else X
            W = self.parameters[f"W{l}"]
            Z = cache[f"Z{l}"]
            
            if self.activation_functions[l-1] == 'relu':
                dZ = dAL * (Z > 0)
            elif self.activation_functions[l-1] == 'sigmoid':
                dZ = dAL * Activation.sigmoid(Z) * (1 - Activation.sigmoid(Z))
            elif self.activation_functions[l-1] == 'tanh':
                dZ = dAL * (1 - np.tanh(Z) ** 2)
            else:
                dZ = dAL  # For linear and softmax, handled separately

            grads[f"dW{l}"] = np.dot(dZ, A_prev.T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            
            dAL = np.dot(W.T, dZ)  # Propagate the gradient backward

        return grads
