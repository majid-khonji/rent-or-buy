import numpy as np
import input as i

# assuming delta T = 1
def DPOA(ins: i.Instance, w):
    ins.w = w
    for t in np.arange(ins.D):
        val = ins.predict(t)
        D_ = val if val != None else t + w*ins.B
        if D_ >= ins.B or t >= ins.B:
            return t + ins.B
    return t+1


def DPOA_eval(ins: i.Instance, w):
    ins.w = w
    for t in np.arange(ins.D):
        val = ins.predict(t)
        D_ = val if val != None else t + w*ins.B
        if D_ >= ins.B or t >= ins.B:
            return 1
#         else:
#             return 0
    return 0


# assuming delta T = 1
def RPOA(ins: i.Instance, w):
    ins.w = w

    upper = (1-w)*ins.B
    # gamma_vals = np.arange(upper)
    # dis = [np.exp(g/upper)/((np.exp(1)-1) * upper) for g in gamma_vals]
    # We need to normalize for some reason
    # dis = dis/sum(dis)
    # gen a random gamma
    # gamma = np.random.choice(gamma_vals, 1,p= dis)

    # using CDF
    gamma = upper * np.log(np.random.uniform() * (np.exp(1) - 1) + 1)
    for t in np.arange(ins.D):
        # print("t = ", t)
        val = ins.predict(t)
        D_ = val if val != None else t + w*ins.B
        if D_ >= gamma + w*ins.B or t >= gamma+w*ins.B:
            # print("Buy...")
            return t + ins.B
        # else:
            # print("keep renting..")
    return t+1

def RPOA_eval(ins: i.Instance, w):
    ins.w = w
    upper = (1-w)*ins.B
    gamma = upper * np.log(np.random.uniform() * (np.exp(1) - 1) + 1)
    for t in np.arange(ins.D):
        val = ins.predict(t)
        D_ = val if val != None else t + w*ins.B
        if D_ >= gamma + w*ins.B or t >= gamma+w*ins.B:
            # print("Buy...")
            return 1
#         else:
#             return 0
    return 0


def OPT(ins: i.Instance):
    if ins.D < ins.B:
        return ins.D
    else:
        return ins.B
    



