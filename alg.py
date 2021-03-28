import numpy as np
import input as i
import alg_google as gl

# assuming delta T = 1
def DPOA(ins: i.Instance, w):
    ins.w = w
    for t in np.arange(ins.D):
        val = ins.predict(t)
        D_ = val if val != None else t + w*ins.B
        if D_ >= ins.B or t >= ins.B:
            return t + ins.B
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
        D_ = val if val != None else t + w*ins.B
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
def OLPA(ins: i.MultiInstance, e=0.25, alg=DPOA, W=10):
    K = ins.K
    R = np.zeros(shape=(K+1,W))
    w_vals = np.linspace(0, 1, W)
    windows = [] # output windows
    total = 0

    norm_factor = ins.norm_factor if ins.normalize else 1
    for k in np.arange(1, K+1):
        dis = [np.exp(-e / (8 * (ins.B / norm_factor) ** 2) * R[k - 1, w]) for w in np.arange(W)]
        dis = dis/sum(dis)
        w = np.random.choice(np.arange(W), 1,p= dis)
        sol = alg(ins.ins[k], w_vals[w]) / norm_factor
        total += sol

        windows.append(float(w_vals[w]))

        for w in np.arange(W):
            alg_sol = alg(ins.ins[k], w_vals[w]) / norm_factor
            R[k,w] = R[k-1,w] + alg_sol


    sol_min_w = np.min(R[K,:])
    # print("windows: ", windows)
    # print("min w: ", w_vals[np.argmin(R[K,:])])
    # print("R: ", R[k,:])
    return total/K, sol_min_w/K#, windows, w_vals[np.argmin(R[K,:])]

# W: number of samples for w
def MOLPA(ins: i.MultiPredictInstance, e=0.25, alg=DPOA, W=10):
    K = ins.K
    S = ins.S


    w_vals = np.linspace(0, 1, W)
    windows = [] # output windows
    preds = [] # predictors
    R = np.zeros(shape=(K+1,W, S))
    sols = np.zeros(shape=(K+1,W, S))

    norm_factor = ins.norm_factor if ins.normalize else 1

    total = 0

    rho = 0
    for s in np.arange(S):
        for k in np.arange(1, K+1):
            for w in np.arange(W):
                sols[k,w,s] = alg(ins.ins[k,s], w_vals[w])/norm_factor # if not ins.normalize else alg(ins.ins[k,s], w_vals[w]) / ins.norm_factor
                R[k, w] = R[k - 1, w, s] + sols[k,w,s]
                rho = sols[k,w,s]/(ins.B/norm_factor) if sols[k,w,s]/(ins.B/norm_factor) > rho else rho

    keys = [(w,s) for w in np.arange(W) for s in np.arange(S)]
    for k in np.arange(1, K+1):
        dis = [np.exp(-e / (2 * rho**2  * (ins.B / norm_factor) ** 2) * R[k - 1, w, s]) for w in np.arange(W) for s in np.arange(S)]
        dis = dis/sum(dis)
        rand = np.random.choice(np.arange(W*S), 1,p= dis)
        rand = np.int(rand)
        w,s = keys[rand]
        # print("w = %d, s=%d"%(w,s))
        sol = sols[k,w,s]
        total += sol

        windows.append(w_vals[w])
        preds.append(s)

    sol_min = np.min(R[K,:,:])
    return total/K, sol_min/K

# Here the algorithm selects best predictor And best algorithm (chosing between Sid's and Google's algs)
# W: number of samples for w
def MOLPA_mult_algs(ins: i.MultiPredictInstance, e=0.25, det = True, W=10, verbose = False):
    K = ins.K
    S = ins.S


    w_vals = np.linspace(0, 1, W)
    windows = [] # output windows
    preds = [] # selected predictors
    algs = [] # selected algs
    R = np.zeros(shape=(K+1,W, S*2)) # the size of S is now twice as bing, because each S is now ran by an algorithm
    sols = np.zeros(shape=(K+1,W, S*2))

    norm_factor = ins.norm_factor if ins.normalize else 1

    total = 0

    rho = 0
    for s in np.arange(S*2): # predictor algorithm pairs
        if det:
            alg = DPOA if s <= S-1 else gl.DPOA_google
        else:
            alg = RPOA if s <= S-1 else gl.RPOA_google


        for k in np.arange(1, K+1):
            for w in np.arange(W):
                sols[k,w,s] = alg(ins.ins[k,np.mod(s,S)], w_vals[w])/norm_factor # if not ins.normalize else alg(ins.ins[k,s], w_vals[w]) / ins.norm_factor
                R[k, w] = R[k - 1, w, s] + sols[k,w,s]
                rho = sols[k,w,s]/(ins.B/norm_factor) if sols[k,w,s]/(ins.B/norm_factor) > rho else rho

    keys = [(w,s) for w in np.arange(W) for s in np.arange(S*2)]
    for k in np.arange(1, K+1):
        dis = [np.exp(-e / (2 * rho**2  * (ins.B / norm_factor) ** 2) * R[k - 1, w, s]) for w in np.arange(W) for s in np.arange(S*2)]
        dis = dis/sum(dis)
        rand = np.random.choice(np.arange(W*S*2), 1,p= dis)
        rand = np.int(rand)
        w,s = keys[rand]
        sol = sols[k,w,s]
        total += sol

        if verbose:
            windows.append(w_vals[w])
            preds.append(np.mod(s,S))
            print("w = %f, s=%d"%(w_vals[w],np.mod(s,S)))
            if s >= S:
                algs.append(1)
                print("Google selected")
            else:
                algs.append(0)
                print("Sid's selected")

    sol_min = np.min(R[K,:,:])
    if verbose:
        return total/K, sol_min/K, algs, preds, windows
    return total/K, sol_min/K#, algs
