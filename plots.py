import numpy as np
import os
import matplotlib.pyplot as plt
from hsic import *
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

unif = (35/255,87/255,137/255)
ada = (217/255,72/255,1/255)

def hist(Y, bins, color="black", alpha=0.3, label=None, scale=None,
         title=None, save_path=None,
         xlabel=None, show=True):
    """
    Histogram with customization options.
    Saves to save_path, if specified.

        Y -- data
        bins -- number of bins
        color -- color of bars
        alpha -- transparency
        label -- label to use for data
        scale -- log or normal
        title -- title of the histogram
        save_path -- where to save the plot. If None, do not save the plot
        xlabel -- label of x axis
        show -- displays the plot or not
    """
    
    if scale == "log":
        hist, bins, _ = plt.hist(Y, bins=bins)
        plt.close()
        if bins[0] == 0:
            bins[0] = bins[1] - 0.99*bins[1]
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(Y, bins=logbins, color=color, alpha=alpha, label=label)
        plt.xscale(scale)
    else:
        plt.hist(Y, bins=bins, color=color, alpha=alpha, label=label)   
    plt.ylabel("N")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else: 
        plt.close()
    

def multi_hist(X, Y, bins, colors, alpha=1, scale=None,
        title=None, save_path=None,
         xlabel=None, show=True):
    """
    Given an input categorical variable X and an output variable Y,
    compare histograms for values of Y conditioned by all possible values of X.
    Saves to save_path, if specified.

        X -- 1D array of input categorical variable
        Y -- 1D-array of corresponding values of output variable
        bins -- number of bins
        color -- list of color of bars, of length len(list(np.unique(X)))
        alpha -- transparency
        label -- list of possible values for X (np.unique(X))
        scale -- log or normal
        title -- title of the histogram
        save_path -- where to save the plot. If None, do not save the plot
        xlabel -- label of x axis
        show -- displays the plot or not
    """
    values = np.unique(X)
    Y0 = [Y[np.where(X == v)] for v in values]
    Y0.append(Y)
    color_hist = [colors[k] for k in range(values.shape[0])]
    color_hist.append("gray")
    label_hist = list(values)
    label_hist.append("all")
    hist(Y0, bins=bins, color=color_hist, alpha=alpha, label=label_hist, scale=scale,
         title=title, save_path=save_path, xlabel=xlabel, show=show)



def hsic_bar(X, N, color, label, var_X=None, width=0.35,
             title=None, save_path=None, show=True):
    """
    Bar visualization of hsics values X for hyperparams.
    Saves to save_path, if specified.

        X -- 1D array of hsics
        N -- number of samples used to compute hsics
        color -- list of color of bars, of length len(X)
        label -- list of hyperparams
        var_X -- variance of hsic estimation
        width -- width of the bar
        title -- title of the plot
        save_path -- where to save the plot. If None, do not save the plot
        show -- displays the plot or not
    """
    X = np.array(X)
    label = np.array(label)
    color = np.array(color)
    order = X.argsort()
    X = X[order]
    label = label[np.flip(order)]
    length = len(X)
    for i in range(X.shape[0]):
        if i == length:
            plt.bar(0, X[length - 1 - i], width, color=color[i%2], label=label[i])
        else:
            plt.bar(0, X[length - 1 - i], width, bottom=np.sum(X[:length - 1 - i]), color=color[i%2], label=label[i])

    x2 = np.linspace(0 - width / 2, 0 + width / 2, X.shape[0])
    if var_X is not None:
        plt.errorbar(x2, np.cumsum(X), yerr=np.sqrt(var_X)/np.sqrt(N), fmt='.', color='black')
    plt.xlim([-0.75, 0.25])
    plt.ylim([0, np.sum(X) + 0.1*np.sum(X)])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) 
    plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

#%%
def scatter_plot(X, info, label, raw=False, smoothing_method="silverman",
                 title=None, save_path=None, show=True):
    """
    Scatter plot of X. If X is 2D, displays a matrix of scatter plots. 
    In the diagonal, computes histograms with kde smoothing.
    Saves to save_path, if specified.

        X -- Array of data
        label -- variables names
        smoothing_method -- smoothing method used for kde in diagonals
        title -- title of the plot
        save_path -- where to save the plot. If None, do not save the plot
        show -- displays the plot or not

        ~~~ specific to HO ~~~
        info -- hyperparam type, if raw=True
        raw -- if the hyperparams are normalized or not
    """

    base_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 7})
    if raw:
        fig = plt.figure(figsize=(35, 27))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                plt.subplot2grid((X.shape[1], X.shape[1]), (i, j))
                if (info[i] == "integer") & (info[j] == "integer"):
                    plt.scatter(np.array(X[:, j], dtype="int"),
                                np.array(X[:, i], dtype="int"))
                elif (info[i] == "continuous") & (info[j] == "continuous"):
                    plt.scatter(np.array(X[:, j], dtype="float"),
                                np.array(X[:, i], dtype="float"))
                elif (info[j] == "continuous") & (info[i] == "integer"):
                    plt.scatter(np.array(X[:, j], dtype="float"),
                                np.array(X[:, i], dtype="int"))
                elif (info[i] == "continuous") & (info[j] == "integer"):
                    plt.scatter(np.array(X[:, j], dtype="int"),
                                np.array(X[:, i], dtype="float"))
                elif (info[i] == "continuous") & (info[j] == "categorical"):
                    plt.scatter(X[:, j],
                                np.array(X[:, i], dtype="float"))
                elif (info[j] == "continuous") & (info[i] == "categorical"):
                    plt.scatter(np.array(X[:, j], dtype="float"),
                                X[:, i])
                elif (info[i] == "integer") & (info[j] == "categorical"):
                    plt.scatter(X[:, j],
                                np.array(X[:, i], dtype="int"))
                elif (info[j] == "integer") & (info[i] == "categorical"):
                    plt.scatter(np.array(X[:, j], dtype="int"),
                                X[:, i])
                if i == X.shape[1] - 1:
                    plt.xlabel(label[j])
                if j == 0:
                    plt.ylabel(label[i])
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    else:
        X_norm = np.array(X, dtype="float")

        fig = plt.figure(figsize=(35, 27))
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):

                if i != j:
                    plt.subplot2grid((X.shape[1], X.shape[1]), (i, j))
                    plt.scatter(X_norm[:, j], X_norm[:, i], s=3, alpha=0.8)
                    plt.xticks([])
                    plt.yticks([])
                    if i == X.shape[1] - 1:
                        plt.xlabel(label[j])
                    if j == 0:
                        plt.ylabel(label[i])
                elif i == j :
                    plt.subplot2grid((X.shape[1], X.shape[1]), (i, j))
                    a = plt.hist(X_norm[:, j], bins=15, label=label[i], density=True)
                    kde(X_norm[:, j], [0,1], gaussian_kernel, 1, method=smoothing_method)
                    plt.ylim([0,np.max(a[0]) + 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.legend()
                    if i == X.shape[1] - 1:
                        plt.xlabel(label[j])
                    if j == 0:
                        plt.ylabel(label[i])



        plt.title(title)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        plt.rcParams.update({'font.size': base_font})

def gaussian_kernel(x, h):
    return 1 / (np.sqrt(2*np.pi) * h) * np.exp(-x ** 2 / (2 * h ** 2))


def kde(X, bounds, kernel, n_params, method="silverman", color="orange"):
    """
    KDE plot of data X. This function works but is clumsy. Has to be reworked.

        X -- 1D array of data
        bounds -- [min, max]
        method -- method used to obtain bandwidth. If silverman or scott, uses 
                            rules of thumb and gaussian kernel. Else, use likelihood estimation
        kernel -- kernel to use

    Returns

        the bandwidth (not very useful)
    """

    def ll(bw):
        est = []
        for i in range(X.shape[0]):
            est.append(np.log(np.mean(kernel(X[i] - X[np.where(X != X[i])], bw))))
        return -np.sum(est)
    if method in ["silverman", "scott"]:
        g_kde = gaussian_kde(X, bw_method=method)
        def fun(X,h):
            return g_kde(X)
        h = "thumb"
    else:
        def fun(x, h):
            return np.mean(kernel(x - X, h))  #
        #h = minimize_scalar(ll, bounds=(0.0001, 5), method="bounded")["x"]
        x0 = np.reshape(np.median(X, axis=0), (n_params,))
        h_bounds = [(0.0001, 5) for i in range(n_params)]
        h = minimize(ll, x0=x0, bounds=h_bounds, method="SLSQP")["x"]
    x = np.linspace(bounds[0], bounds[1], 1000)
    y = np.array([fun(xi, h) for xi in x])
    plt.plot(x, y, color=color)
    return h

def parallel_plot(X, Y, q, infos, labels, normalized=True,
                 title=None, save_path=None, show=True):
    """
    Parralel plot of X, with X.shape[1] variables in axis. Only 
    X of quantile q are displayed. Otherwise it would be equually distributed.
    Saves to save_path, if specified.

        X -- 2D array of input variables
        Y -- 1D array of data
        infos -- list of data type ("categorical", "continuous", "continuous_log"
                 or "integer")
        infos -- list of labels of input variables
        title -- title of the plot
        save_path -- where to save the plot. If None, do not save the plot
        show -- displays the plot or not
    """
    base_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 10})
    X_norm = np.array(X)
    if not normalized:
        for i in range(X.shape[1]):
            X_norm[:,i] = col_comparable(X[:,i], infos[i])
    q_Y = np.quantile(Y, q, interpolation='lower')
    Y_l = np.log(Y)
    Y_l -= np.min(Y_l)
    Y_ln = np.array(Y_l) / np.max(np.abs(Y_l))
    p_plot = np.c_[X_norm, Y_ln]
    label_pp = list(labels)
    label_pp.append("loss")

    plt.figure(figsize=(30, 10))
    p_plot = np.array(p_plot, dtype="float")
    p_plot_worst = p_plot[np.where(Y > q_Y)[0], :]
    for plot in p_plot_worst:
        plt.plot(label_pp, plot, 'o', color="gold", alpha=0.05)
    for plot in p_plot_worst:
        plt.plot(label_pp, plot, color="gold", alpha=0.05)

    p_plot_best = p_plot[np.where(Y <= q_Y)[0], :]
    for plot in p_plot_best:
        plt.plot(label_pp, plot, 'o', color="purple", alpha=0.2)
    for plot in p_plot_best:
        plt.plot(label_pp, plot, color="purple", alpha=0.2)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    plt.rcParams.update({'font.size': base_font})


def col_comparable(X, info):
    """
    Tool function for parallel plot. Enforce variables to be between [0,1],
    without necessarily making them uniform.
    """
    if info == "categorical":
        values = np.unique(X)
        n = values.shape[0]
        X = [np.where(x == values)[0][0]/(n-1) for x in X]
        X = np.array(X, dtype='float')
    elif info in ["integer", "continuous"]:
        X = np.array(X, dtype='float')
        X = X - np.min(X)
        X = X / np.max(X)
        X = [x for x in X]
        X = np.array(X)
    elif info in ["continuous_log"]:
        X = -np.log(np.array(X, dtype='float'))
        X = X - np.min(X)
        X = X / np.max(X)
        X = [x for x in X]
        X = np.array(X)
    return X

