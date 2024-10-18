import numpy as np

class Activation:
    @staticmethod
    def linear(Z):
        return Z

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
