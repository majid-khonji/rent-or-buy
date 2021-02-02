import numpy as np
import math
import input as i

def DPOA_google(ins: i.Instance, w):
    ins.w = w
    _lambda = 1 - w
    D_ = ins.noisy_D
    if D_ >= ins.B:
        if ins.D < math.ceil(ins.B * _lambda):
            return ins.D
        else:
            return math.ceil(_lambda * ins.B) + ins.B - 1
    else:
        if ins.D < math.ceil(ins.B / _lambda):
            return ins.D
        else:
            return math.ceil(ins.B / _lambda) + ins.B - 1


def RPOA_google(ins: i.Instance, w):
    ins.w = w
    _lambda = 1 - w
    if _lambda < 1 / ins.B:
        _lambda = 1 / ins.B
    D_ = ins.noisy_D
    if D_ >= ins.B:
        k = math.floor(_lambda * ins.B)
        k = max(1,k)
        i_vals = np.arange(k)
        dis = [math.pow((ins.B - 1) / ins.B, k - m) / (ins.B * (1 - math.pow(1 - 1 / ins.B, k))) for m in i_vals]
        _sum = sum(dis)
        _dis = [x / _sum for x in dis]
        j = np.random.choice(i_vals, 1, p=_dis)
        j = max(1,j)
        if ins.D < j:
            return ins.D
        else:
            return j + ins.B - 1
    else:
        l = math.ceil(ins.B / _lambda)
        l = max(1,l)
        i_vals = np.arange(l)
        dis = [math.pow((ins.B - 1) / ins.B, l - m) / (ins.B * (1 - math.pow(1 - 1 / ins.B, l))) for m in i_vals]
        _sum = sum(dis)
        _dis = [x / _sum for x in dis]
        j = np.random.choice(i_vals, 1, p=_dis)
        j = max(1,j)

        if ins.D < j:
            return ins.D
        else:
            return j + ins.B - 1
