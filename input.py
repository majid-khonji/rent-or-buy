import numpy as np
class Instance:
    T = 100 # max duration
    D = None # resource duration
    B = 50 # price to buy
    def __init__(self, B=60):
        self.D = np.random.randint(1, self.T)
        self.B = B
    def needed(self, t):
        return t < self.D

class Predictor:
    ins:Instance = None
    w = .5 # prediction horizon
    def predict(self, t):
        pass

class PerfectPredictor(Predictor):
    def __init__(self, ins:Instance, w = None):
        self.ins = ins
        self.w = w if w!= None else None
    def predict(self, t):
        if t <= self.ins.D <= t + self.w * self.ins.B: #>= self.ins.B:
            return self.ins.D
        else:
            return None



class NoisyPredictor(Predictor):
    def __init__(self, ins:Instance, w = None, std = 25):
        self.ins = ins
        self.w = w if w!= None else None
        self.std = std
    def predict(self, t):
        noisy_D  = np.random.normal(loc = self.ins.D, scale = self.std)
        # print("noisy D: ", noisy_D)
        if t <= noisy_D <= t + self.w * self.ins.B:
            return self.ins.D
        else:
            return None
