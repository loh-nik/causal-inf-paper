import numpy as np
import oct2py
from oct2py import octave
#oc = oct2py.Oct2Py()
octave.addpath("./StateSpaceGC")

def significanceMatrix(G, model_ord, samples, alpha):
    from scipy.stats import chi2
    d = G.shape[0]
    pVals = 1 - chi2.cdf(G, df = model_ord, loc=0, scale = 1/samples)
    for i in range(d):
        pVals[i,i] = 1
    # bonferroni multiple hypothesis test was decided against in order to avoid skewing results on density (where more edges need to be identified as significant)
    m = d**2
    q = (1 /(d*(d-1))) *alpha
    flatPValues = pVals.flatten()
    argSorted = np.argsort(flatPValues)
    inverseSort = np.argsort(argSorted)
    sortedPVals = flatPValues[argSorted]
    significanceArray = np.zeros(m)
    sortedSignificanceMatrix = np.zeros(m)
    for i in range(m):
        # reject null hypothesis of insignificance if our p value is small enough
        significanceArray[i] = sortedPVals[i] < (i+1)*q
    sortedSignificanceMatrix = significanceArray[inverseSort]

    sortedSignificanceMatrix = sortedSignificanceMatrix.reshape(G.shape)
    #return sortedSignificanceMatrix
    return pVals < alpha

# X in shape (variables, observations)
def gcss(X, alpha, tau_max, returnAll = False):
    q, N = X.shape
    _, pbic = octave.ar_IC(X, tau_max, False, nout=2, verbose=False)
    pbic = max(pbic, tau_max)
    if pbic != tau_max:
        print("GCSS pbic order greater than given tau_max")
    # fix model order to num_variables * tau_max 
    m,A,C,K,V = octave.s4sid_CCA(X, pbic, int(pbic*q), nout=5,verbose=False)
    G = octave.iss_PWGC(A,C,K,V, nout=1,verbose=False)
    # in theory, the degrees of freedom of the chi^2 distribution should be the model order
    # we keep it at the number of variables q to maintain acceptable detection rates for larger tau_max
    signif = significanceMatrix(G, q, N, alpha)
    if returnAll:
        return np.array(G)
    return np.array(G)*signif

def close_octave():
    if octave:
        try:
            octave.exit()
        except Exception as e:
            print(f"Failed to close Oct2Py: {e}")

# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     plt.plot(chi2.cdf(np.arange(99) * 0.01, df=3, loc=0, scale = 1/1000))
#     plt.figure()
#     plt.plot(chi2.ppf(np.arange(99) * 0.01, df=3, loc=0, scale = 1/1000))
#     print(1 - chi2.cdf(0.021, df=12, loc=0, scale = 1/1000))
#     print(chi2.ppf(0.95,df=3, loc=0, scale = 1/1000))
#     plt.show()