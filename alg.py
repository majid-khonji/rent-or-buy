import numpy as np
import input as i

# assuming delta T = 1
def DPOA(ins: i.Instance, pred:i.Predictor, w):
    pred.w = w
    for t in np.arange(ins.D):
        # print("t = ", t)
        val = pred.predict(t)
        if val != None:
            D_ = val
            # print("D_ = ", D_)
        else:
            D_ = t + w*ins.B
        if D_ >= ins.B or t >= ins.B:
            # print("Buy...")
            return t + ins.B
        # else:
            # print("keep renting..")
    return t+1

def OPT(ins: i.Instance):
    if ins.D < ins.B:
        return ins.D
    else:
        return ins.B



