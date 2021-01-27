import numpy as np
class Instance:
    T = 100 # max duration
    D = None # resource duration
    B = 50 # price to buy
    noisy_D = 0 # read by the predictor
    w = 0 # predictor window
    def __init__(self, w=0.1, B=50, predictor_std = 0):
        self.D = np.random.randint(1, self.T)
        self.B = B
        self.w = w
        if predictor_std != 0:
            self.noisy_D = int(np.random.normal(loc=self.D, scale=predictor_std))
        else:
            self.noisy_D = self.D

    def needed(self, t):
        return t < self.D

    def predict(self, t):
        if t <= self.noisy_D <= t + self.w * self.B:  # >= self.ins.B:
            return self.noisy_D
        else:
            return None

class MultiInstance:
    B = 50
    ins = []
    K= None

    # K: overrides the upper bound of k in theorem 3, None will take the default value
    def __init__(self, K=10, B=50, predictor_std=0):
        self.B = B
        if K == None:
            theta = 2 * np.log(12 * self.B ** 2 / np.exp(1))
            self.K = int(np.ceil(12 * self.B ** 2 * theta / np.exp(2)))
        else:
            self.K = K

        self.ins = [Instance(B=self.B, predictor_std=predictor_std) for _ in np.arange(self.K + 1)]


