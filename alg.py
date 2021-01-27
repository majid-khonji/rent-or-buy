import numpy as np
import input as i

# assuming delta T = 1
def DPOA(ins: i.Instance, w):
    ins.w = w
    for t in np.arange(ins.D):
        # print("t = ", t)
        val = ins.predict(t)
        if val != None:
            D_ = val
        else:
            D_ = t + w*ins.B
        if D_ >= ins.B or t >= ins.B:
            # print("Buy...")
            return t + ins.B
        # else:
            # print("keep renting..")
    return t+1

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
        if val != None:
            D_ = val
        else:
            D_ = t + w*ins.B

        if D_ >= gamma + w*ins.B or t >= gamma+w*ins.B:
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



# W: number of samples for w
def OLPA(m_ins: i.MultiInstance, e=0.25, alg=DPOA, W=100):
    K = m_ins.K
    R = np.zeros(shape=(K+1,W))
    w_vals = np.linspace(0, 1, W)
    windows = [] # output windows
    sol = 0
    for k in np.arange(1, K+1):
        dis = [np.exp(-e/(8*m_ins.B**2) * R[k-1, w] ) for w in np.arange(W)]
        dis = dis/sum(dis)
        w = np.random.choice(np.arange(W), 1,p= dis)
        sol = sol + alg(m_ins.ins[k], w_vals[w])

        windows.append(w_vals[w])

        for w in np.arange(W):
            R[k,w] = R[k-1,w] + alg(m_ins.ins[k], w_vals[w])


    sol_min_w = np.min(R[K,:])
    return sol/K, sol_min_w/K

