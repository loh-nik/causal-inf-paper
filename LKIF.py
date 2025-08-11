import numpy as np
from LK_Info_Flow import multi_causality_est
from scipy.stats import norm

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

# X in shape (variables, observations)
def lkif(X,alpha, returnAll = False, tau_max = 1, timestamps=None, returnRaw = False):
    IF_result=multi_causality_est(X, max_lag=tau_max, np=1, dt=1, series_temporal_order=timestamps, significance_test=1)
    if returnAll:
        return absmaxND(IF_result['nIF'],axis=2).T
    elif returnRaw:
        return absmaxND(IF_result['IF'], axis = 2).T
    return absmaxND(IF_result['nIF']* (np.abs(IF_result['IF']) - (norm.ppf(1-(alpha/2)) * IF_result['SEIF']) > 0), axis=2).T

# X in shape (variables, observations)
def lkif(X,alpha, returnAll = False, tau_max = 1, timestamps=None, returnRaw = False, returnSignif = False, eulerForward=1):
    IF_result=multi_causality_est(X, max_lag=tau_max, np=eulerForward, dt=1, series_temporal_order=timestamps, significance_test=1)
    m,n = X.shape
    #flat = IF_result['nIF'].flatten(order='C')
    #reshaped = np.reshape(flat, (3,4,3), order="F").transpose(1,0,2)
    nif = IF_result['nIF'].flatten(order='C').reshape((m,tau_max,m), order="F").transpose(1,0,2)
    IF_r = IF_result['IF'].flatten(order='C').reshape((m,tau_max,m), order="F").transpose(1,0,2)
    seif = IF_result['SEIF'].flatten(order='C').reshape((m,tau_max,m), order="F").transpose(1,0,2)
    if returnAll:
        if returnSignif:
            return absmaxND(nif, axis=0).T, absmaxND(nif* (np.abs(IF_r) - (norm.ppf(1-(alpha/2)) * seif) > 0), axis=0).T
        return absmaxND(nif,axis=0).T
    elif returnRaw:
        return absmaxND(IF_r, axis = 0).T
    return absmaxND(nif* (np.abs(IF_r) - (norm.ppf(1-(alpha/2)) * seif) > 0), axis=0).T