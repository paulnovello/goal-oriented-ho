import numpy as np
from scipy.optimize import minimize_scalar

def rbf(x, bw):
    return np.exp(-np.sum(x**2, axis=-1)/(2*bw**2))

def distr_norm(X, info, bounds):
    """
    Normalizes the hyperparams to project them into [0,1] and simulate uniform distr.

        X -- 1D array of raw hyperparam values
        info -- type of hyperparam ("categorical", "continuous", "integer", 
                "continuous_log")
        bounds -- bounds of hyperparam, on the form [min, max]. If info = "categorical",
                  bounds is the list of possible values.

    Returns

        X -- 1D array of normalized hyperparam values
    """

    if info == "categorical":
        values = np.array(bounds)
        n = values.shape[0]
        X = [np.random.random_sample()*1/n + np.where(values==x)[0][0]/n for x in X]
        X = np.array(X, dtype='float')
    elif info == "integer":
        X = np.array(X, dtype='float')
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = X - lw_bnd
        X = X / (up_bnd - lw_bnd)
        values = (np.arange(lw_bnd, up_bnd, 1) - lw_bnd) / (up_bnd - lw_bnd)
        values = np.array(values, dtype='float')
        incr = np.min(np.abs(values[:-1] - values[1:]))
        X = X / (1 + incr)
        incr /= (1 + incr)
        X = [np.random.random_sample()*incr + x for x in X]
        X = np.array(X)
    elif info == "continuous":
        X = np.array(X, dtype='float')
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = (X - lw_bnd)/(up_bnd - lw_bnd)
    elif info == "continuous_log":
        X = np.array(X, dtype='float')
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = - (np.log(X) - np.log(up_bnd)) / (np.log(up_bnd) - np.log(lw_bnd)) 
    else:
        print("info incorrect")
    return X

def reverse_distr_norm(X, info, bounds):
    """
    Transform a normalized hyperparam back to its realistic, raw value

        X -- 1D array of normalized hyperparam values
        info -- string, type of hyperparam ("categorical", "continuous", "integer", 
                "continuous_log")
        bounds -- bounds of hyperparam, on the form [min, max]. If info = "categorical",
                  bounds is the list of possible values.

    Returns

        X -- 1D array of raw hyperparam values
    """

    X = np.array(X, dtype="float")
    if info == "categorical":
        n = len(bounds)
        val = X*n
        val = np.array(val, dtype="int")
        X = bounds[np.min([val, n-1])]
    elif info == "integer":
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = (up_bnd - lw_bnd)*X + lw_bnd
        X = np.round(X)
        X = np.array(X, dtype="int")
    elif info == "continuous":
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = (up_bnd - lw_bnd)*X + lw_bnd
    elif info == "continuous_log":
        up_bnd = float(bounds[1])
        lw_bnd = float(bounds[0])
        X = np.exp(-X * (np.log(up_bnd) - np.log(lw_bnd)) + np.log(up_bnd))
    else:
        print("info incorrect")
    return X

def normalize_data(X, infos, bounds):
    """
    Applies distr_norm() to all the hyperparams at once.

        X -- 2D array of raw hyperparams values
        info -- list of type of hyperparams ("categorical", "continuous", "integer", 
                "continuous_log"). 
        bounds -- list of bounds of hyperparam, on the form [min, max]. 
                  If info = "categorical", bounds is the list of possible values.

    Returns:

        X_norm -- 2D array of normalized hyperparams values
    """

    X_norm = np.array(X)
    for i in range(X.shape[1]):
        X_norm[:, i] = distr_norm(X[:, i], infos[i], bounds[i])
    X_norm = np.array(X_norm, dtype='float')
    return X_norm

def reverse_normalize_data(X, infos, bounds):
    """
    Applies reverse_distr_norm() to all the hyperparams at once.

        X -- 2D array of normalized hyperparams values
        info -- list of type of hyperparams ("categorical", "continuous", "integer", 
                "continuous_log"). 
        bounds -- list of bounds of hyperparam, on the form [min, max]. 
                  If info = "categorical", bounds is the list of possible values.

    Returns:

        X -- 2D array of raw hyperparams values
    """

    X_norm = np.array(X)
    for i in range(X_norm.shape[1]):
        X_norm[:, i] = reverse_distr_norm(X[:, i], infos[i], bounds[i])
    X = np.array(X_norm, dtype='float')
    return X

def hsic(x, Y, p, raw=False, info=None, bounds=None, bw="median"):
    """
    Computes the hsic of input variable x, given output variable Y for quantile p.
    The remaining arguments will certainly be removed since they are manages by 
    new src.ho_postproc.HOPostproc class. 

        x -- array of samples of input variable. Can be 2D for multidimensionnal variable,
             used for interaction hsics.
        Y -- 1D-array of corresponding values of output variable
        p -- quantile to consider
        bw -- string, the bandwidth method used to compute the bandwidth of the 
              RKHS kernel (here gaussian). If set to a value, this value is used.
              "max" or "median".
        ~~~ unused, likely to be removed, because specific to HO ~~~
        raw -- boolean, if the hyperparam values are raw or normalized. If raw, apply distr_norm().
        info -- string, hyperparam type
        bounds -- hyperparam bounds, of the form [min, max]

    Returns

        [s, v] with
        s -- hsic of input variable
        v -- variance of estimation of s     
    """

    X = np.array(x)
    n = X.shape[0]
    if len(X.shape) == 1:
        X = np.reshape(X, (n,1))
    l = X.shape[1]

    ###### unused
    if raw:
        if l == 1:
            X[:, 0] = distr_norm(X[:, 0], info, bounds)
        else:
            for i in range(l):
                X[:,i] = distr_norm(X[:,i], info[i], bounds[i])
        X = np.array(X, dtype="float")

    ###### create Z, the indicatrice variable
    thres = np.quantile(Y, p, interpolation='lower')
    Z = np.array(X)
    Z[np.where(Y > thres)[0],:] = 0
    Zr = Z[np.where(Y <= thres)[0],:]

    ###### creates matrices for estimating S
    m = Zr.shape[0]
    XX_ini = X[:,:1] - X[:,:1].T
    XZ_ini = X[:,:1] - Zr[:,:1].T
    ZZ_ini = Zr[:,:1] - Zr[:,:1].T
    XX_ini = np.reshape(XX_ini, (n, n, 1))
    XZ_ini = np.reshape(XZ_ini, (n, m, 1))
    ZZ_ini = np.reshape(ZZ_ini, (m, m, 1))
    for i in range(1,l):
        XX_ini = np.append(XX_ini, np.reshape(X[:, i:i+1] - X[:, i:i+1].T, (n,n,1)), axis=-1)
        XZ_ini = np.append(XZ_ini, np.reshape(X[:, i:i+1] - Zr[:, i:i+1].T, (n,m,1)), axis=-1)
        ZZ_ini = np.append(ZZ_ini, np.reshape(Zr[:, i:i+1] - Zr[:, i:i+1].T, (m,m,1)), axis=-1)
    
    ###### creates MMD function to estimate hsic, with bandwidth as argument,
    ###### to be able to select bandwidth depending of the selection method.
    def MMD(bw):
        XX = rbf(XX_ini, bw=bw)
        XZ = rbf(XZ_ini, bw=bw) 
        ZZ = rbf(ZZ_ini, bw=bw)
        mean = p**2 * (np.mean(XX) + np.mean(ZZ) - 2 * np.mean(XZ))
        var = p**4 * (1/(n**2 - 1)*np.sum((XX - np.mean(XX))**2)) + \
              p**4 * (1/(m**2 - 1)*np.sum((ZZ - np.mean(ZZ))**2))  + \
              p**4 * (4/(m*n - 1)*np.sum((XZ - np.mean(XZ))**2))
        return mean, var
                
                #(np.var(XX) - 2 * np.var(XZ) + np.var(ZZ))

    ###### three ways of selecting the bandwidth
    if bw == "median":
        bw_final = np.median(np.abs(XX_ini))
    elif bw == "max":
        def neg_MMD(bw):
            return -MMD(bw)[0]
        bw_final = minimize_scalar(neg_MMD, bounds=(0.0001, 5), method="bounded")["x"]
    else:
        bw_final = bw
    return MMD(bw_final)

