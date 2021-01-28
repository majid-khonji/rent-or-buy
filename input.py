import numpy as np
class Instance:
    T = 100 # max duration
    D = None # resource duration
    B = 50 # price to buy
    noisy_D = 0 # read by the predictor
    w = 0 # predictor window
    def __init__(self, w=0.1, B=50, predictor_std = 0, D_dependant = False):
        self.D = np.random.randint(1, self.T)
        self.B = B
        self.w = w
        if predictor_std != 0:
            noisy_D = int(np.random.normal(loc=self.D, scale=predictor_std))
            self.noisy_D =  max(0, noisy_D)
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
    T = 100
    B = 50
    ins = []
    K= None
    normalize = False
    norm_factor = T

    # K: overrides the upper bound of k in theorem 3, None will take the default value
    # normalize: used to devide sol values by norm_factor
    def __init__(self, K=10, B=50, predictor_std=0, normalize = True):
        self.B = B
        self.normalize = normalize
        self.K = K
        self.ins = [Instance(B=self.B, predictor_std=predictor_std) for _ in np.arange(self.K + 1)]


class MultiPredictInstance:
    B = 50
    ins = None # 2d array[s,k]
    K = None # number of epochs
    S = None # number of predictors


    # K: overrides the upper bound of k in theorem 3, None will take the default value
    def __init__(self,S=5, K=10, B=50, predictor_std_range=[0,100]):
        self.B = B
        self.S = S
        self.K = K

        self.ins = np.empty(shape=(S,K+1), dtype=Instance)
        std_list = np.linspace(predictor_std_range[0], predictor_std_range[1], S)
        for s in np.arange(S):
            for k in np.arange(K+1):
                self.ins[s,k] = Instance(B=self.B, predictor_std=std_list[s])
