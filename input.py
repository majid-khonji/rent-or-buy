import numpy as np
class Instance:
    T = 100 # max duration
    D = None # resource duration
    B = 50 # price to buy
    noisy_D = 0 # read by the predictor
    time_dependant = False
    predictor_std = 0 # stores initial predictor std
    def __init__(self, w=0.1, B=50, predictor_std = 0, time_dependant = False):
        self.D = np.random.randint(1, self.T)
        self.B = B
        self.w = w
        self.time_dependant = time_dependant
        self.predictor_std = predictor_std
        self.time_dependant = time_dependant

        if predictor_std == 0:
            self.noisy_D = self.D
        else:
            noisy_D = int(np.random.normal(loc=self.D, scale=self.predictor_std))
            self.noisy_D =  max(0, noisy_D)



    def needed(self, t):
        return t < self.D

    def predict(self, t):
        if self.predictor_std != 0 and not self.time_dependant:
            noisy_D = int(np.random.normal(loc=self.D, scale=self.predictor_std))
            noisy_D =  max(0, noisy_D)
        elif self.time_dependant:
            # print("std ", self.predictor_std*(self.D - t)/self.D)
            noisy_D = int(np.random.normal(loc=self.D, scale= self.predictor_std*(self.D - t)/self.D))
            noisy_D = max(0, noisy_D)
        else:
            noisy_D = self.D
        # add prediction randomness here
        if t <= noisy_D <= t + self.w * self.B:  # >= self.ins.B:
            return noisy_D
        else:
            return None

class MultiInstance:
    T = 100
    B = 50
    ins = []
    K= None
    normalize = False
    norm_factor = T
    w = 0.1

    # K: overrides the upper bound of k in theorem 3, None will take the default value
    # normalize: used to devide sol values by norm_factor
    def __init__(self, K=10, B=50,  predictor_std=0, normalize = True, k_dependant = False, time_dependant = False):
        self.B = B
        self.normalize = normalize
        self.K = K
        if k_dependant:
            self.ins = [Instance(B=self.B, predictor_std=predictor_std*(K-k)/K, time_dependant=time_dependant) for k in np.arange(self.K + 1)]
        else:
            self.ins = [Instance(B=self.B, predictor_std=predictor_std, time_dependant=time_dependant) for _ in np.arange(self.K + 1)]
    def to_t_dependant(self):
        for k in range(self.K + 1):
            self.ins[k].time_dependant = True


class MultiPredictInstance:
    T = 100
    B = 50
    ins = None # 2d array[s,k]
    K = None # number of epochs
    S = None # number of predictors
    normalize = False
    norm_factor = T
    w = 0.1

    # K: overrides the upper bound of k in theorem 3, None will take the default value
    def __init__(self,S=5, K=10, w=0.1, B=50, predictor_std_range=[0,100], normalize = True):
        self.B = B
        self.S = S
        self.K = K
        self.normalize = normalize
        self.w = w

        self.ins = np.empty(shape=(K+1,S), dtype=Instance)
        std_list = np.linspace(predictor_std_range[0], predictor_std_range[1], S)
        for s in np.arange(S):
            for k in np.arange(K+1):
                self.ins[k,s] = Instance(B=self.B, predictor_std=std_list[s])
