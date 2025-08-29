import numpy as np
import DataGenerator

import Visualization as vis
#from matplotlib import pyplot as plt
from progress.bar import Bar
import time
#import argparse

# these matrices are used often so we just define them globally
mediumCouplingMatrixCascade_LowDense = np.array([
                                        [0,0,0,0,0,0],
                                        [1,0,0,0,0,0],
                                        [0,-1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,0,0,1],
                                        [0,0,0,0,-1,0]])
mediumCouplingMatrixVAR_LowDense = np.array([
                                    [0.5,0,0,0,0,0],
                                    [1,0.5,0,0,0,0],
                                    [0,-1,0.5,0,0,0],
                                    [0,0,0,0.5,0,0],
                                    [0,0,1,0,0.5,1],
                                    [0,0,0,0,-1,0.5]])

mediumCouplingMatrixVAR_HighDense = np.array([
                                    [0.5,0,-1,-1,0,0],
                                    [1,0.5,0,0,0,0],
                                    [1,1,0.5,0,0,0],
                                    [0,0,0,0.5,-1,-1],
                                    [0,1,-1,0,0.5,1],
                                    [0,0,0,1,-1,0.5]])
mediumCouplingMatrixCascade_HighDense = np.array([
                                    [0,0,-1,-1,0,0],
                                    [1,0,0,0,0,0],
                                    [1,1,0,0,0,0],
                                    [0,0,0,0,-1,-1],
                                    [0,1,-1,0,0,1],
                                    [0,0,0,1,-1,0]])

defaultCouplingMatrixVAR_LowDense= np.array([[0.5,0,0],[-1,0.5,0],[0,-1,0.5]])
defaultCouplingMatrixCascade_LowDense = np.array([[0,0,0],[-1,0,0],[0,-1,0]])

defaultCouplingMatrixVAR_HighDense= np.array([[0.5,1,0],[-1,0.5,1],[1,-1,0.5]])
defaultCouplingMatrixCascade_HighDense = np.array([[0,1,0],[-1,0,1],[1,-1,0]])

largeCouplingMatrixVAR_LowDense = np.array([[0.5,0,0,0,0,0,0,0,0,0,0,0],
                                    [1,0.5,0,0,0,0,0,0,0,0,0,0],
                                    [0,1,0.5,0,0,0,0,0,0,0,0,0],
                                    [0,1,0,0.5,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0.5,-1,0,0,0,0,0,0],
                                    [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                    [0,0,-1,0,0,0,0.5,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0.5,1,0,0,0],
                                    [0,0,0,0,0,0,0,-1,0.5,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0.5,0,-1],
                                    [0,0,0,0,0,0,0,-1,0,0,0.5,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0.5]])
largeCouplingMatrixCascade_LowDense = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                                    [1,0,0,0,0,0,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,-1,0,0,0,0,0,0],
                                    [0,0,0,-1,0,0,0,0,0,0,0,0],
                                    [0,0,-1,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,0,0,-1,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,0,-1],
                                    [0,0,0,0,0,0,0,-1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0,0]])
largeCouplingMatrixVAR_HighDense = np.array([[0.5,0,-1,0,0,1,0,0,0,0,0,0],
                                    [1,0.5,-1,0,0,0,0,0,0,0,0,0],
                                    [0,1,0.5,0,0,0,0,1,0,0,0,0],
                                    [0,1,0,0.5,0,0,0,-1,0,0,0,0],
                                    [0,0,0,0,0.5,-1,0,0,-1,-1,0,0],
                                    [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                    [0,0,-1,0,0,1,0.5,0,0,0,0,0],
                                    [0,0,0,0,0,0,-1,0.5,1,0,0,0],
                                    [0,0,0,0,0,0,0,-1,0.5,1,0,0],
                                    [0,0,0,0,1,0,0,0,0,0.5,1,-1],
                                    [0,0,0,0,0,0,0,-1,0,0,0.5,1],
                                    [0,0,0,1,0,0,0,0,0,0,0,0.5]])
largeCouplingMatrixCascade_HighDense = np.array([[0,0,-1,0,0,1,0,0,0,0,0,0],
                                    [1,0,-1,0,0,0,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0,1,0,0,0,0],
                                    [0,1,0,0,0,0,0,-1,0,0,0,0],
                                    [0,0,0,0,0,-1,0,0,-1,-1,0,0],
                                    [0,0,0,-1,0,0,0,0,0,0,0,0],
                                    [0,0,-1,0,0,1,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,-1,0,1,0,0,0],
                                    [0,0,0,0,0,0,0,-1,0,1,0,0],
                                    [0,0,0,0,1,0,0,0,0,0,1,-1],
                                    [0,0,0,0,0,0,0,-1,0,0,0,1],
                                    [0,0,0,1,0,0,0,0,0,0,0,0]])

# get label encoding the full ranking of 3 algorithms
def getLabel(scoreTriplet):
    gcss, lkif, pcm = scoreTriplet
    if gcss > lkif:
        if pcm > gcss:
            #return "PCMCIGCSSLKIF"
            return 0
        elif pcm > lkif:
            #return "GCSSPCMCILKIF"
            return 1
        else:
            #return "GCSSLKIFPCMCI"
            return 2
    else:
        if pcm > lkif:
            #return "PCMCILKIFGCSS"
            return 3
        elif pcm > gcss:
            #return "LKIFPCMCIGCSS"
            return 4
        else:
            #return "LKIFGCSSPCMCI"
            return 5
        
# get a label that is the highest scoring algorithm for easier visual representation
def getMaxLabel(scoreTriplet):
    gcss, lkif, pcm = scoreTriplet
    if gcss == lkif and lkif == pcm: return 3
    if gcss > lkif:
        if gcss > pcm: return 0
        elif gcss == pcm: return 3
        else: return 2
    elif pcm > lkif: return 2
    elif pcm == lkif or gcss == lkif: return 3
    else: return 1

def getMeanStdDev(data, axis = None):
    return np.average(data, axis = axis), np.std(data, axis = axis)

def getMedianQuantile(data, quantile, axis=None):
    return np.median(data, axis=axis), np.quantile(data, q = quantile/2, axis=axis), np.quantile(data, q=1-(quantile/2), axis=axis)

def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def maxSignificantLink(x, bool_matr, axis = None):
    y = np.multiply(x, bool_matr)
    return absmaxND(y, axis = axis)

def getFullMetrics(a,b):
    a = np.abs(a) > 0
    b = np.abs(b) > 0
    #convert bool to int
    a = a*1
    b = b*1
    truePos = np.sum(a+b == 2)
    # diagonals are always true negatives, subtract them
    trueNeg = np.sum(a+b == 0) - a.shape[0]
    falsePos = np.sum(np.maximum(b-a,0))
    falseNeg = np.sum(np.maximum(a-b,0))
    return [truePos, trueNeg, falsePos, falseNeg]

def MCC(arr):
    #print(arr)
    truePos, trueNeg, falsePos, falseNeg = arr
    return float(((truePos * trueNeg) - (falsePos * falseNeg)) / np.sqrt((truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg)) if (truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg) != 0 else 0)

def MCCFromFull(a, axis=0):
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(MCC, axis, a)

#with ground truth a, result b
def tpr_fpr_Scores(a,b):
    a = np.abs(a) > 0
    b = np.abs(b) > 0
    #convert bool to int
    a = a*1
    b = b*1
    totalPos = np.sum(a)
    # remove diagonal from negative
    totalNeg = a.shape[0]*a.shape[0] - totalPos - a.shape[0]
    truePos = np.sum(a+b == 2)
    falsePos = np.sum(np.maximum(b-a,0))
    tpr = truePos / totalPos
    fpr = falsePos / totalNeg
    return [tpr,fpr]

def tpr_fpr(arr):
    truePos, trueNeg, falsePos, falseNeg = arr
    return [float(truePos / (truePos + falseNeg)), float(falsePos / (falsePos + trueNeg))]

def tpr_fpr_FromFull(a, axis=0):
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(tpr_fpr, axis, a)

# returns a numpy array of shape (2, len(algorithms))
# or shape (4, len(algorithms)) for evalType "full" 
def getMetricOfRealization(couplingMatrix, algorithms, model, samples, alpha, couplingStrength, noiseScale, tauMax, seed, deltaTCascadeOutput, evalType, delayLength = 0, fullData = [], returnMatrices = False):
    import GCSS
    import LKIF
    import PCMCI
    tauList = False
    if type(tauMax) is list:
        if len(tauMax) != len(algorithms):
            print("Error: length of tauMax must equal number of selected algorithms")
            return
        tauList = True
    # dont change autoregressive components
    matrix = couplingStrength * couplingMatrix
    for i in range(matrix.shape[0]):
        matrix[i,i] = couplingMatrix[i,i]
    if model == "Cascade":
        fullData = np.array(DataGenerator.getCascadeDataBrainpy(matrix, samples, delayLength)).T
    elif model == "VAR":
        fullData = np.array(DataGenerator.getVARData(matrix, samples, noiseScale, seed, delayLength))
    matrices = []
    # fullData has the shape (observations, variables), LKIF and GCSS want that transposed
    if "GCSS" in algorithms:
        tauGCSS = tauMax
        if tauList: 
            tauGCSS = tauMax[algorithms.index("GCSS")]
        #matrixGCSS = GCSS.gcss(fullData.T, alpha, tauMax, returnAll=False)
        try:
            matrixGCSS = GCSS.gcss(fullData.T, alpha, tauGCSS, returnAll=False)
            matrixGCSS[np.isnan(matrixGCSS)] = 0
        except:
            matrixGCSS = np.zeros((fullData.shape[1],fullData.shape[1]))
            print("Error on GCSS")
        np.fill_diagonal(matrixGCSS,0)
        matrices.append(matrixGCSS)
    if "LKIF" in algorithms:
        tauLKIF = tauMax
        if tauList: 
            tauLKIF = tauMax[algorithms.index("LKIF")]
        matrixLKIF = LKIF.lkif(fullData.T, alpha, tau_max=tauLKIF)
        np.fill_diagonal(matrixLKIF, 0)
        matrices.append(matrixLKIF)
    if "PCMCI" in algorithms:
        tauPCMCI = tauMax
        if tauList: 
            tauPCMCI = tauMax[algorithms.index("PCMCI")]
        matrixPCMCI, p_values = PCMCI.PCMCIPlus(fullData, [], range(fullData.shape[1]), None, tauPCMCI, alpha if alpha<=1 else 1, contempLinks=True)
        graph_bool = p_values <= alpha
        matrixPCMCI = maxSignificantLink(matrixPCMCI, graph_bool, axis = 2)
        np.fill_diagonal(matrixPCMCI,0)
        matrices.append(matrixPCMCI.T)
    np.fill_diagonal(couplingMatrix, 0)
    if returnMatrices:
        return np.array(matrices)
    if evalType == "ROC":
        metrics = np.array([tpr_fpr_Scores(couplingMatrix, matrix) for matrix in matrices])
    elif evalType == "Full":
        metrics = np.array([getFullMetrics(couplingMatrix, matrix) for matrix in matrices])
    else:
        print("Error: Invalid evaluation type found")
    return metrics.T

def delay6dEvaluations(plotOnly, delaySizes, samples, alpha, randomRuns, couplStrength):
    if not plotOnly:
        # this data should have: 3 variables, 100 runs, 2000 samples per run
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = []
        for j in range(len(delaySizes)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplStrength, noiseScale= 0.01, tauMax= int(delaySizes[j] * 10) + 1,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full", delayLength=delaySizes[j])
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 80% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, delaySizes, "", "./diagrams/delays6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay (no unit)", ylabel ="MCC")
    vis.saveMCCCurve(median.T, delaySizes, "", "./diagrams/delays6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay (no unit)", ylabel ="MCC")

    # VAR systems with additional delay, note that 1 time step delay is always in there due to time discretization, so we add between 1 and 30 time steps to that
    if not plotOnly:
        seed = 0
        truthMatrix = mediumCouplingMatrixVAR_LowDense
        fullOut = []
        for j in range(len(delaySizes)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplStrength * 0.1, noiseScale= 0.01, tauMax= int(delaySizes[j] * 10) + 1,
                                                  seed= seed, deltaTCascadeOutput= 0, evalType= "Full", delayLength=delaySizes[j] * 10)
                seed += 1
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/VAR_6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/VAR_6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 80% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, [int(d*10) for d in delaySizes], "", "./diagrams/delays6dVAR", stdDev.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay in time steps", ylabel ="MCC")
    vis.saveMCCCurve(median.T, delaySizes, "", "./diagrams/delays6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay in time steps", ylabel ="MCC")

def sample6dEvaluations(plotOnly, sampleCounts, alpha, randomRuns, tauMax, couplingStrength):
    if not plotOnly:
        # this data should have: 3 variables, 100 runs, 2000 samples per run
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = []
        for j in range(len(sampleCounts)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                 samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full")
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/6d_Samples_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/6d_Samples_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 80% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, sampleCounts, "", "./diagrams/samples6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Amount of Samples", ylabel ="MCC", xscale ="log")
    vis.saveMCCCurve(median.T, sampleCounts, "", "./diagrams/samples6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Amount of Samples", ylabel ="MCC",xscale ="log")

    # VAR model
    if not plotOnly:
        seed = 0
        truthMatrix = mediumCouplingMatrixVAR_LowDense
        fullOut = []
        for j in range(len(sampleCounts)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength*0.1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= seed, deltaTCascadeOutput= 0, evalType= "Full")
                seed += 1
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/VAR_6d_Samples_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/VAR_6d_Samples_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 80% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, sampleCounts, "", "./diagrams/samples6dVAR", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Amount of Samples", ylabel ="MCC", xscale ="log")
    vis.saveMCCCurve(median.T, sampleCounts, "", "./diagrams/samples6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Amount of Samples", ylabel ="MCC")

def couplStrength6dEvaluations(plotOnly, couplStrengths, samples, alpha, randomRuns, tauMax):
    if not plotOnly:
        # this data should have: 3 variables, 100 runs, 2000 samples per run
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = []
        for j in range(len(couplStrengths)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                 samples = samples, alpha= alpha, couplingStrength=  couplStrengths[j] * 10, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full")
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 90% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, couplStrengths*10, "", "./diagrams/coupl6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="MCC")
    vis.saveMCCCurve(median.T, couplStrengths*10, "", "./diagrams/coupl6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="MCC")

    if not plotOnly:
        seed = 0
        truthMatrix = mediumCouplingMatrixVAR_LowDense
        fullOut = []
        for j in range(len(couplStrengths)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplStrengths[j], noiseScale= 0.01, tauMax= tauMax,
                                                  seed= seed, deltaTCascadeOutput= 0, evalType= "Full")
                seed += 1
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/VAR_6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/VAR_6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    # get central 90% of data
    median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
    vis.saveMCCCurve(mean.T, couplStrengths, "", "./diagrams/coupl6dVAR", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="MCC")
    vis.saveMCCCurve(median.T, couplStrengths, "", "./diagrams/coupl6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="MCC")

def scale_lightness(rgb, scale_l):
    import colorsys
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def system6dEvaluations(plotOnly, alpha, samples, randomRuns, tauMax, couplingStrength):
    import matplotlib as mpl
    defaultCols = mpl.color_sequences["tab10"]
    # low density: n-1 edges
    # high density: 2n (or 2n-1 for 3 nodes)
    cascMatrices = [defaultCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade_LowDense, largeCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade_HighDense, mediumCouplingMatrixCascade_HighDense, largeCouplingMatrixCascade_HighDense]
    VARMatrices = [defaultCouplingMatrixVAR_LowDense, mediumCouplingMatrixVAR_LowDense, largeCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR_HighDense, mediumCouplingMatrixVAR_HighDense, largeCouplingMatrixVAR_HighDense]
    
    if not plotOnly:
        fullOut = []
        for j in range(len(cascMatrices)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = cascMatrices[j], algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full")
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/6d_System_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/6d_System_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    lowDense = mean[:3]
    highDense = mean[3:]
    final = np.append(lowDense, highDense, axis=1)

    # get central 90% of data
    #median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)

    vis.saveMCCCurve(final[:,::3].T, [3,6,12], "", "./diagrams/SystemCascGCSS", [], show=False, save = True, rowLabels=["GCSS-LD", "GCSS-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    vis.saveMCCCurve(final[:,1::3].T, [3,6,12], "", "./diagrams/SystemCascLKIF", [], show=False, save = True, rowLabels=["LKIF-LD", "LKIF-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    vis.saveMCCCurve(final[:,2::3].T, [3,6,12], "", "./diagrams/SystemCascPCMCI", [], show=False, save = True, rowLabels=["PCMCI-LD","PCMCI-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    
    vis.saveMCCCurve(final.T, [3,6,12], "", "./diagrams/SystemCasc", [], show=False, save = True, rowLabels=["GCSS-LD", "LKIF-LD", "PCMCI-LD","GCSS-HD", "LKIF-HD", "PCMCI-HD"], 
                     colors=[defaultCols[0], defaultCols[1], defaultCols[2], scale_lightness(defaultCols[0], 1.6), scale_lightness(defaultCols[1], 1.6), scale_lightness(defaultCols[2], 1.6)], xlabel = "Variable Count", ylabel ="MCC")
    #vis.saveMCCCurve(median.T, delaySizes, "", "./diagrams/delays6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay (no unit)", ylabel ="MCC")

    if not plotOnly:
        seed = 0
        fullOut = []
        for j in range(len(VARMatrices)):
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(couplingMatrix = VARMatrices[j], algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplingStrength*0.1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= seed, deltaTCascadeOutput= 0, evalType= "Full")
                seed += 1
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        np.save("./data/VAR_6d_System_"+str(randomRuns)+"_runs_metrics.npy", fullOut)

    else: fullOut = np.load("./data/VAR_6d_System_"+str(randomRuns)+"_runs_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    lowDense = mean[:3]
    highDense = mean[3:]
    final = np.append(lowDense, highDense, axis=1)
    # get central 90% of data
    #median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)

    # decide how to display the results
    vis.saveMCCCurve(final.T, [3,6,12], "", "./diagrams/SystemVAR", [], show=False, save = True, rowLabels=["GCSS-LD", "LKIF-LD", "PCMCI-LD","GCSS-HD", "LKIF-HD", "PCMCI-HD"],
                     colors=[defaultCols[0], defaultCols[1], defaultCols[2], scale_lightness(defaultCols[0], 1.6), scale_lightness(defaultCols[1], 1.6), scale_lightness(defaultCols[2], 1.6)], xlabel = "Variable Count", ylabel ="MCC")
    vis.saveMCCCurve(final[:,::3].T, [3,6,12], "", "./diagrams/SystemVARGCSS", [], show=False, save = True, rowLabels=["GCSS-LD", "GCSS-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    vis.saveMCCCurve(final[:,1::3].T, [3,6,12], "", "./diagrams/SystemVARLKIF", [], show=False, save = True, rowLabels=["LKIF-LD", "LKIF-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    vis.saveMCCCurve(final[:,2::3].T, [3,6,12], "", "./diagrams/SystemVARPCMCI", [], show=False, save = True, rowLabels=["PCMCI-LD","PCMCI-HD"], 
                     xlabel = "Variable Count", ylabel ="MCC")
    #vis.saveMCCCurve(median.T, delaySizes, "", "./diagrams/delays6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay in time steps", ylabel ="MCC")

def nonStationaryTipping():
    loadData = False
    if not loadData:
        #dT = lambda x,t : max(min((-0.4 / 20) + 0.8, 0), -0.4)
        dT = lambda x,t : (t >= 30) * (t <= 70) * (-0.4 / 40)
        #truthMatrix = np.array([[0,1,0],[-1,0,0],[0,0,0]])
        #truthMatrix = np.array([[0,0,0],[0,0,1],[0,-1,0]])
        #truthMatrix = np.array([[0,1], [-1,0]])
        #truthMatrix = np.array([[0,0], [-1,0]])
        truthMatrix = np.array([[0,0,0,0,0,0],
                                        [1,0,0,0,0,0],
                                        [0,-1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,0,0,1],
                                        [0,0,0,0,-1,0]])
        dim = 6
        randomRuns = 10
        metricsNormal = []
        metricsReducedInfo = []
        for i in range(randomRuns):
            data = DataGenerator.getCascade6dConfoundedBrainpy(dT, truthMatrix, 1000)
            #plt.plot(data.T)
            #plt.show()
            #exit()
            matrices = getMetricOfRealization(truthMatrix, ["GCSS", "LKIF", "PCMCI"], "None", 1000, 0.05, 0, 0, [5,1,5], 0, 0, "Full", fullData = data.T, returnMatrices=True)
            vis.saveCouplingMatrixGraph(matrices[0], "", "diagrams/gcssFullInfo", False, True)
            vis.saveCouplingMatrixGraph(matrices[1], "", "diagrams/lkifFullInfo", False, True)
            vis.saveCouplingMatrixGraph(matrices[2], "", "diagrams/pcmciFullInfo", False, True)
            matrices = matrices[:,:dim,:dim]
            metrics = np.array([getFullMetrics(truthMatrix, matrix) for matrix in matrices])
            metricsNormal.append(metrics.T)
            reducedMetrics = getMetricOfRealization(truthMatrix, ["GCSS", "LKIF", "PCMCI"], "None", 1000, 0.05, 0, 0, [5,1,5], 0, 0, "Full", fullData = data[:dim].T, returnMatrices = True)
            vis.saveCouplingMatrixGraph(reducedMetrics[0], "", "diagrams/gcssNoInfo", False, True)
            vis.saveCouplingMatrixGraph(reducedMetrics[1], "", "diagrams/lkifNoInfo", False, True)
            vis.saveCouplingMatrixGraph(reducedMetrics[2], "", "diagrams/pcmciNoInfo", False, True)
            reducedMetrics = np.array([getFullMetrics(truthMatrix, matrix) for matrix in reducedMetrics])
            metricsReducedInfo.append(reducedMetrics.T)
        np.save("./data/Casc_fullInfo6d.npy", metricsNormal)
        np.save("./data/Casc_reducedInfo6d.npy", metricsReducedInfo)
    else: 
        metricsNormal = np.load("./data/Casc_fullInfo6d.npy")
        metricsReducedInfo = np.load("./data/Casc_reducedInfo6d.npy")
    metricsNormal = np.array(metricsNormal)
    metricsReducedInfo=np.array(metricsReducedInfo)
    print(metricsNormal.shape)
    print(metricsReducedInfo.shape)
    print(metricsNormal[:,:,2])
    print(metricsReducedInfo[:,:,2])
    fullInfoMCC = MCCFromFull(metricsNormal, axis=1)
    reducedInfoMCC = MCCFromFull(metricsReducedInfo, axis=1)
    fullInfoMean, fullInfoStd = getMeanStdDev(fullInfoMCC, axis = 0)
    reducedInfoMean, reducedInfoStd = getMeanStdDev(reducedInfoMCC, axis=0)
    print(fullInfoMean)
    print(fullInfoStd)
    print(reducedInfoMean)
    print(reducedInfoStd)

def nonStationaryStable(plotOnly, ceilings, alpha, samples, tauMax, randomRuns):
    """We increase forcing linearly over the course of 50 units of time (500 samples), up to some ceiling, then compare performances across ceiling heights.
    Experiment only conducted for nonlinear system, as there's no tipping in the VAR system"""
    
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        dim = 6
        metricsNormal = np.zeros((len(ceilings), randomRuns, 4, 3))
        metricsReducedInfo = np.zeros((len(ceilings), randomRuns, 4, 3))
        tipped = np.zeros(len(ceilings))
        for i in range(len(ceilings)):
            for j in range(randomRuns):
                dT = lambda x,t : (t <= 50) * (-0.4 / 50) * ceilings[i]
                data = DataGenerator.getCascade6dConfoundedBrainpy(dT, truthMatrix, samples, 0.001)
                # if we cross the threshold between +1 and -1, we assume tipped
                if np.min(data[:dim]) < 0: tipped[i] += 1
                matrices = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "None", 
                                                 samples = samples, alpha= alpha, couplingStrength= 1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full", fullData = data.T, returnMatrices=True)
                matrices = matrices[:,:dim,:dim]
                metricsNormal[i,j] = np.array([getFullMetrics(truthMatrix, matrix) for matrix in matrices]).T
                matrices2 = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "None", 
                                                 samples = samples, alpha= alpha, couplingStrength= 1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= 0, deltaTCascadeOutput= 0, evalType= "Full", fullData = data[:dim].T, returnMatrices=True)
                metricsReducedInfo[i,j] = np.array([getFullMetrics(truthMatrix, matrix) for matrix in matrices2]).T
            tipped[i] = tipped[i] / randomRuns
        np.save("./data/Casc_forcing6d.npy", metricsNormal)
        np.save("./data/Casc_forcing_reducedInfo_6d.npy", metricsReducedInfo)
        np.save("./data/Casc_forcing6d_tippedFraction.npy", tipped)
    else: 
        metricsNormal = np.load("./data/Casc_forcing6d.npy")
        metricsReducedInfo = np.load("./data/Casc_forcing_reducedInfo_6d.npy")
        tipped = np.load("./data/Casc_forcing6d_tippedFraction.npy")
    
    vis.saveMCCCurve(tipped, ceilings, "", "./diagrams/forcing6d_tippedFraction", xlabel="Forcing Strength", ylabel="Fraction of runs with tipping")

    fullInfoMCC = MCCFromFull(np.array(metricsNormal), axis=2)
    fullInfoMean, fullInfoStd = getMeanStdDev(fullInfoMCC, axis = 1)
    noInfoMCC = MCCFromFull(np.array(metricsReducedInfo), axis=2)
    noInfoMean, noInfoStd = getMeanStdDev(noInfoMCC, axis = 1)

    gcssInfo = np.stack((fullInfoMean[:,0], noInfoMean[:,0]), axis=1)
    lkifInfo = np.stack((fullInfoMean[:,1], noInfoMean[:,1]), axis=1)
    pcmInfo = np.stack((fullInfoMean[:,2], noInfoMean[:,2]), axis=1)
    gcssStd = np.stack((fullInfoStd[:,0], noInfoStd[:,0]), axis=1)
    lkifStd = np.stack((fullInfoStd[:,1], noInfoStd[:,1]), axis=1)
    pcmStd = np.stack((fullInfoStd[:,2], noInfoStd[:,2]), axis=1)
    
    vis.saveMCCCurve(fullInfoMean.T, ceilings, "", "./diagrams/forcing6d", fullInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
    vis.saveMCCCurve(noInfoMean.T, ceilings, "", "./diagrams/forcing6d_reducedInformation", noInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
    
    vis.saveMCCScatter(gcssInfo.T, ceilings, "", "./diagrams/forcing6d_gcss", gcssStd.T, rowLabels=["Known Confounder", "Hidden Confounder"],figsize=(8,2), dpi=300)
    vis.saveMCCScatter(lkifInfo.T, ceilings, "", "./diagrams/forcing6d_lkif", lkifStd.T, rowLabels=["Known Confounder", "Hidden Confounder"],figsize=(8,2), dpi=300)
    vis.saveMCCScatter(pcmInfo.T, ceilings, "", "./diagrams/forcing6d_pcmci", pcmStd.T, rowLabels=["Known Confounder", "Hidden Confounder"],figsize=(8,2), dpi=300)
    
    fullInfoMCC = tpr_fpr_FromFull(np.array(metricsNormal), axis=2)
    fullInfoMean, fullInfoStd = getMeanStdDev(fullInfoMCC, axis = 1)
    noInfoMCC = tpr_fpr_FromFull(np.array(metricsReducedInfo), axis=2)
    noInfoMean, noInfoStd = getMeanStdDev(noInfoMCC, axis = 1)

    gcssInfo = np.append(fullInfoMean[:,:,0], noInfoMean[:,:,0], axis=1)
    lkifInfo = np.append(fullInfoMean[:,:,1], noInfoMean[:,:,1], axis=1)
    pcmInfo = np.append(fullInfoMean[:,:,2], noInfoMean[:,:,2], axis=1)
    gcssStd = np.append(fullInfoStd[:,:,0], noInfoStd[:,:,0], axis=1)
    lkifStd = np.append(fullInfoStd[:,:,1], noInfoStd[:,:,1], axis=1)
    pcmStd = np.append(fullInfoStd[:,:,2], noInfoStd[:,:,2], axis=1)

    print(gcssInfo.shape)
    
    # vis.saveMCCCurve(fullInfoMean.T, ceilings, "", "./diagrams/forcing6d_TPR", fullInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
    # vis.saveMCCCurve(noInfoMean.T, ceilings, "", "./diagrams/forcing6d_reducedInformation_TPR", noInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
    
    vis.saveMCCScatter(gcssInfo.T, ceilings, "", "./diagrams/forcing6d_gcss_TPR", gcssStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR", "Hidden Confounder TPR", "Hidden Confounder FPR"],figsize=(8,2), dpi=300)
    vis.saveMCCScatter(lkifInfo.T, ceilings, "", "./diagrams/forcing6d_lkif_TPR", lkifStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR","Hidden Confounder TPR",  "Hidden Confounder FPR"],figsize=(8,2), dpi=300)
    vis.saveMCCScatter(pcmInfo.T, ceilings, "", "./diagrams/forcing6d_pcmci_TPR", pcmStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR","Hidden Confounder TPR",  "Hidden Confounder FPR"],figsize=(8,2), dpi=300)
    

def autoCorrEvaluations():
    loadCasc = False
    autoCorrelations = [0.1, 0.25, 0.5, 0.75, 1, 1.25,1.5,2,4]
    samples = 1000
    alpha =0.05
    if not loadCasc:
        data = np.load("./data/3d_multiCorr_20_runs_1000_samples.npy")
        print(data.shape)
        
        # this data should have: 3 variables, 100 runs, 2000 samples per run
        truthMatrix = np.array([[0,1,0],[-1,0,0],[1,-1,0]])
        fullOut = []
        for j in range(data.shape[0]):
            output = []
            for i in range(data.shape[1]):
                metrics = getMetricOfRealization(truthMatrix, ["GCSS", "LKIF", "PCMCI"], "None", samples, alpha, 0, 0, 5, 0, 0, "Full", fullData = data[j,i].T)
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        print(fullOut.shape)
        #print(fullOut)
        np.save("./data/3d_multiCorr_20_runs_1000_samples_metrics.npy", fullOut)
    else: 
        fullOut = np.load("./data/3d_multiCorr_20_runs_1000_samples_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    
    vis.saveMCCCurve(mean.T, autoCorrelations, "", "./diagrams/autoCorrCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Mean Reversion Rate", ylabel ="MCC")

    print("Cascade data finished")

    # VAR systems with additional delay, note that 1 time step delay is always in there due to time discretization, so we add between 1 and 30 time steps to that
    loadVAR = False
    randomRuns = 20
    autoCorrelations = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    if not loadVAR:
        seed = 0
        
        fullOut = []
        for j in range(len(autoCorrelations)):
            crr = autoCorrelations[j]
            truthMatrix = np.array([[crr,1,0],[-1,crr,0],[1,-1,crr]])
            output = []
            for i in range(randomRuns):
                metrics = getMetricOfRealization(truthMatrix, ["GCSS", "LKIF", "PCMCI"], "VAR", samples, alpha, 0.1, 0.01, 5, seed, 0, "Full")
                seed += 1
                output.append(metrics)
            fullOut.append(output)
        fullOut = np.array(fullOut)
        print(fullOut.shape)
        np.save("./data/VAR_3d_multiCorr_20_runs_1000_samples_metrics.npy", fullOut)

    else: fullOut = np.load("./data/VAR_3d_multiCorr_20_runs_1000_samples_metrics.npy")
    scores = MCCFromFull(fullOut, axis=2)
    mean, stdDev = getMeanStdDev(scores, axis = 1)
    
    vis.saveMCCCurve(mean.T, autoCorrelations, "", "./diagrams/autoCorrVAR", stdDev.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Autocorrelation per time step", ylabel ="MCC")

def mainEvaluations(analysisIndex = None):

    startTime = time.process_time()

    folder = "./MainExperiment/"

    
    # since a and b in the x^3 equations make up the autoregressive component, we replace it by 0.5 for var systems
    defaultCouplingMatrixVAR= np.array([[0.5,1,0],[-1,0.5,0],[1,-1,0.5]])
    defaultCouplingMatrixCascade = np.array([[0,1,0],[-1,0,0],[1,-1,0]])
    defaultSamples = 500
    defaultCouplingStrengthVAR = 0.1
    defaultCouplingStrengthCascade = 1
    defaultNoiseScale = 0.1
    defaultTauMax = 2

        
    # larger systems, similar density - about 1.3 edges per vertex
    mediumCouplingMatrixVAR = np.array([[0.5,0,-1,1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [0,1,0.5,0,0,0],
                                        [0,0,0,0.5,0,0],
                                        [0,0,1,-1,0.5,1],
                                        [0,0,0,0,-1,0.5]])
    mediumCouplingMatrixCascade = np.array([[0,0,-1,1,0,0],
                                        [1,0,0,0,0,0],
                                        [0,1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,-1,0,1],
                                        [0,0,0,0,-1,0]])
            
    largeCouplingMatrixVAR = np.array([[0.5,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0.5,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade = np.array([[0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0]])
    
    mediumCouplingMatrixCascade_LowDense = np.array([
                                        [0,0,0,0,0,0],
                                        [1,0,0,0,0,0],
                                        [0,-1,0,0,0,0],
                                        [0,0,0,0,0,0],
                                        [0,0,1,0,0,1],
                                        [0,0,0,0,-1,0]])
    mediumCouplingMatrixVAR_LowDense = np.array([
                                        [0.5,0,0,0,0,0],
                                        [1,0.5,0,0,0,0],
                                        [0,-1,0.5,0,0,0],
                                        [0,0,0,0.5,0,0],
                                        [0,0,1,0,0.5,1],
                                        [0,0,0,0,-1,0.5]])
    
    mediumCouplingMatrixVAR_HighDense = np.array([
                                        [0.5,0,-1,-1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [1,1,0.5,0,0,0],
                                        [0,0,0,0.5,-1,-1],
                                        [0,1,-1,0,0.5,1],
                                        [0,0,0,1,-1,0.5]])
    mediumCouplingMatrixCascade_HighDense = np.array([
                                        [0,0,-1,-1,0,0],
                                        [1,0,0,0,0,0],
                                        [1,1,0,0,0,0],
                                        [0,0,0,0,-1,-1],
                                        [0,1,-1,0,0,1],
                                        [0,0,0,1,-1,0]])
    
    defaultCouplingMatrixVAR_LowDense= np.array([[0.5,0,0],[-1,0.5,0],[0,-1,0.5]])
    defaultCouplingMatrixCascade_LowDense = np.array([[0,0,0],[-1,0,0],[0,-1,0]])

    defaultCouplingMatrixVAR_HighDense= np.array([[0.5,1,0],[-1,0.5,1],[1,-1,0.5]])
    defaultCouplingMatrixCascade_HighDense = np.array([[0,1,0],[-1,0,1],[1,-1,0]])

    largeCouplingMatrixVAR_LowDense = np.array([[0.5,0,0,0,0,0,0,0,0,0,0,0],
                                        [1,0.5,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,0,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade_LowDense = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                                        [1,0,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,0,0,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,0,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,0,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,0,0,0,0,0,0,0]])
    largeCouplingMatrixVAR_HighDense = np.array([[0.5,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0.5,-1,0,0,0,0,0,0,0,0,0],
                                        [0,1,0.5,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0.5,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0.5,-1,0,0,-1,-1,0,0],
                                        [0,0,0,-1,0,0.5,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0.5,0,0,0,0,0],
                                        [0,0,0,0,0,0,-1,0.5,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0.5,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0.5,1,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0.5,1],
                                        [0,0,0,1,0,0,0,0,0,0,0,0.5]])
    largeCouplingMatrixCascade_HighDense = np.array([[0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [1,0,-1,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,1,0,0,0,0],
                                        [0,1,0,0,0,0,0,-1,0,0,0,0],
                                        [0,0,0,0,0,-1,0,0,-1,-1,0,0],
                                        [0,0,0,-1,0,0,0,0,0,0,0,0],
                                        [0,0,-1,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,0,0,-1,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,-1,0,1,0,0],
                                        [0,0,0,0,1,0,0,0,0,0,1,-1],
                                        [0,0,0,0,0,0,0,-1,0,0,0,1],
                                        [0,0,0,1,0,0,0,0,0,0,0,0]])
    matrixFolder = "./MatrixGraphs/"
    saveCouplingMatrices = True
    checkCouplingMatrices = False
    showCouplingMatrices = False
    if checkCouplingMatrices:
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR, "", matrixFolder+"SmallMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_HighDense, "", matrixFolder+"SmallHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(defaultCouplingMatrixVAR_LowDense, "", matrixFolder+"SmallLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR, "", matrixFolder+"MedMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_HighDense, "", matrixFolder+"MedHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(mediumCouplingMatrixVAR_LowDense, "", matrixFolder+"MedLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR, "", matrixFolder+"LargeMedDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_HighDense, "", matrixFolder+"LargeHighDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)
        vis.saveCouplingMatrixGraph(largeCouplingMatrixVAR_LowDense, "", matrixFolder+"LargeLowDense", showCouplingMatrices, saveCouplingMatrices, dpi=200)

    runsPerConfigAlpha = 10
    runsPerConfigTauMax = 20
    runsPerConfigSeparateParams = 20
    runsPerConfigCombinedParams = 10
    runsPerConfigAlphaCombination = 10
    runsPerConfigAlphaBonus = 20
    doSave = True
    showDiagrams = False
    # to test that data shapes are correctly transmitted everywhere
    fakeRun = False

    # the first boolean indicates default behavior when no arguments is passed for analysisIndex
    plotAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 1
    calculateAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 1
    alphaSizeFileName = "alphaSizeAnalysis.npy"
    alphaDensityFileName = "alphaDensityAnalysis.npy"

    plotTauMax = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 2
    calculateTauMax = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 2
    tauMaxFileName = "tauMaxAnalysis.npy"

    plotSeparateDataParameters = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 3
    calculateSeparateDataParameters = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 3
    dataParamsFileName = "dataParameters.npz"

    plotDecisionTree = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 4
    calculateFullSearchSpace = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 4
    fullSearchFileName = "fullSearchSpace.npy"

    plotAlphaCombined = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 5
    calculateAlphaCombined = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 5
    alphaCombinedFileName = "alphaCombination.npy"

    plotBonusAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6
    calculateBonusAlpha = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6
    calculateBonusAlpha2 = (False and analysisIndex == None) or analysisIndex == 0 or analysisIndex == 6

    seed = 0
    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    tauMaxs = [2,3,5,7,10]

    samples_separate = [20,50,100,200,500,1000,2000,5000,10000]
    couplingStrVAR_separate = np.array([0.01, 0.02, 0.05, 0.07,0.1,0.15,0.2,0.25,0.3])
    couplingStrCascade_separate = 10*couplingStrVAR_separate
    noiseScales_separate = [0.01,0.02,0.05,0.1,0.2,0.5]

    matrices_separate_VAR = [defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR, defaultCouplingMatrixVAR_HighDense, 
                            mediumCouplingMatrixVAR_LowDense, mediumCouplingMatrixVAR, mediumCouplingMatrixVAR_HighDense, 
                            largeCouplingMatrixVAR_LowDense, largeCouplingMatrixVAR, largeCouplingMatrixVAR_HighDense]
    matrices_separate_Cascade = [defaultCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade, defaultCouplingMatrixCascade_HighDense,
                                 mediumCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade, mediumCouplingMatrixCascade_HighDense,
                                 largeCouplingMatrixCascade_LowDense, largeCouplingMatrixCascade, largeCouplingMatrixCascade_HighDense]
    
    alphas_combined = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    samples_combined = [100, 500, 1000, 10000]
    couplingStrVAR_combined = np.array([0.01, 0.05, 0.1, 0.3])
    couplingStrCascade_combined = 10*couplingStrVAR_combined
    noiseScales_combined = [0.01,0.05, 0.1, 0.5]
    matrices_combined_VAR = [defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR, defaultCouplingMatrixVAR_HighDense, 
                            mediumCouplingMatrixVAR_LowDense, mediumCouplingMatrixVAR, mediumCouplingMatrixVAR_HighDense]
    matrices_combined_Cascade = [defaultCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade, defaultCouplingMatrixCascade_HighDense,
                                 mediumCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade, mediumCouplingMatrixCascade_HighDense]

    if plotAlpha:
        if calculateAlpha:
            #default analysis for 3 vertices, 4 edges
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")

            seed, resultsMediumSize_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsMediumSize_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLargeSize_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= largeCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLargeSize_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= largeCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLowDense_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR_LowDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLowDense_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade_LowDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")
            
            seed, resultsLargeDense_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixVAR_HighDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed= seed, evalType="Full")
            seed, resultsLargeDense_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= mediumCouplingMatrixCascade_HighDense, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlpha, fakeRun=fakeRun, startSeed=seed, evalType="Full")

            # alpha lies in axis 2
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsMediumSize_VAR = resultsMediumSize_VAR[2]
            resultsMediumSize_Cascade = resultsMediumSize_Cascade[2]
            resultsLargeSize_VAR = resultsLargeSize_VAR[2]
            resultsLargeSize_Cascade = resultsLargeSize_Cascade[2]
            resultsLowDense_VAR = resultsLowDense_VAR[2]
            resultsLowDense_Cascade = resultsLowDense_Cascade[2]
            resultsLargeDense_VAR = resultsLargeDense_VAR[2]
            resultsLargeDense_Cascade = resultsLargeDense_Cascade[2]

            resultsSmall = np.stack((results_VAR, results_Cascade), axis=0)
            resultsMedium = np.stack((resultsMediumSize_VAR, resultsMediumSize_Cascade), axis=0)
            resultsLarge = np.stack((resultsLargeSize_VAR, resultsLargeSize_Cascade), axis = 0)
            resultsLowDense = np.stack((resultsLowDense_VAR,resultsLowDense_Cascade),axis=0)
            resultsHighDense = np.stack((resultsLargeDense_VAR,resultsLargeDense_Cascade), axis=0)
            resultsSize = np.stack((resultsSmall, resultsMedium, resultsLarge),axis=0)
            resultsDense = np.stack((resultsLowDense, resultsMedium, resultsHighDense), axis=0)
            if doSave:
                np.save(folder + alphaSizeFileName, resultsSize)
                np.save(folder+alphaDensityFileName, resultsDense)
        else: 
            resultsSize = np.load(folder + alphaSizeFileName)
            resultsDense = np.load(folder+alphaDensityFileName)
        # shape: 3 (low/medium/large size or density), 2 (VAR/Cascade), 2 (TPR/FPR) or 4 (TP/TN/FP/FN), 3 (Algorithm), runsPerConfig, alpha values
        # average/median over runs
        print(np.squeeze(resultsSize[0,0,:,2,:,10]))
        resultsSize = np.average(tpr_fpr_FromFull(resultsSize, axis= 2), axis= 4)
        resultsDense = np.average(tpr_fpr_FromFull(resultsDense, axis= 2), axis=4)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsSize[0,:,0,0,:], resultsSize[0,:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsSize[0,:,0,1,:], resultsSize[0,:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsSize[0,:,0,2,:], resultsSize[0,:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")

        # for ease of notation: reshape
        resultsSize = np.reshape(resultsSize,(6, resultsSize.shape[2], resultsSize.shape[3],resultsSize.shape[4]),order="F")
        vis.saveROCCurve(resultsSize[:,0,0,:], resultsSize[:,1,0,:], alphas, "GCSS ROC Curve for system sizes", folder+"GCSS_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsSize[:,0,1,:], resultsSize[:,1,1,:], alphas, "LKIF ROC Curve for system sizes", folder+"LKIF_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsSize[:,0,2,:], resultsSize[:,1,2,:], alphas, "PCMCI ROC Curve for system sizes", folder+"PCMCI_Alpha_Size", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        
        # for ease of notation: reshape
        resultsDense = np.reshape(resultsDense,(6, resultsDense.shape[2], resultsDense.shape[3],resultsDense.shape[4]),order="F")
        vis.saveROCCurve(resultsDense[:,0,0,:], resultsDense[:,1,0,:], alphas, "GCSS ROC Curve for system densities", folder+"GCSS_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsDense[:,0,1,:], resultsDense[:,1,1,:], alphas, "LKIF ROC Curve for system densities", folder+"LKIF_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)
        vis.saveROCCurve(resultsDense[:,0,2,:], resultsDense[:,1,2,:], alphas, "PCMCI ROC Curve for system densities", folder+"PCMCI_Alpha_Density", colors=[(0.3,0,0), (0.6,0,0), (0.9,0.2,0.2), (0,0,0.3), (0,0,0.6), (0.2,0.2,0.9)],
                        rowLabels=["VAR Small", "VAR Medium", "VAR Large", "Cascade Small", "Cascade Medium", "Cascade Large"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate", annotateBest=False)

    if plotTauMax:
        if calculateTauMax:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=[],couplingStrengths=[],noiseScales=[],tauMaxs=tauMaxs,
                                            runsPerConfig=runsPerConfigTauMax, fakeRun=fakeRun, startSeed= seed, iterateSeeds=True, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=[],couplingStrengths=[],noiseScales=[],tauMaxs=tauMaxs,
                                            runsPerConfig=runsPerConfigTauMax, fakeRun=fakeRun, startSeed=seed, iterateSeeds=True, evalType="Full")
            #tau max in dim 5
            resultsTauMax = np.stack((results_VAR[5], results_Cascade[5]),axis=0)
            if doSave:
                np.save(folder+  tauMaxFileName, resultsTauMax)
        else:
            resultsTauMax = np.load(folder+tauMaxFileName)
        # shape: 2 (VAR/Cascade), 2 (TPR/FPR) or 4 (TP/TN/FP/FN), 2 (Algorithm), runsPerConfig, tauMax values
        resultsTauMax = np.average(tpr_fpr_FromFull(resultsTauMax, axis=1), axis= 3)
        vis.saveROCCurve(resultsTauMax[:,0,0,:], resultsTauMax[:,1,0,:], tauMaxs, "GCSS ROC Curve for tauMax values", folder+"GCSS_tauMax_ConstSeed_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave)
        vis.saveROCCurve(resultsTauMax[:,0,1,:], resultsTauMax[:,1,1,:], tauMaxs, "PCMCI ROC Curve for tauMax values", folder+"PCMCI_tauMax_ConstSeed_VAR_Cascade", rowLabels=["VAR", "Cascade"], show=showDiagrams, save=doSave)
        
    # uses MCC from here, so matrices are shaped differently!
    if plotSeparateDataParameters:
        if calculateSeparateDataParameters:
            seed, [matricesResults_VAR, sampleResults_VAR, _, couplingStrengthResults_VAR, noiseScaleResults_VAR, _] = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_separate_VAR, samples=samples_separate,alphas=[],couplingStrengths=couplingStrVAR_separate,noiseScales=noiseScales_separate,tauMaxs=[],
                                            runsPerConfig=runsPerConfigSeparateParams, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, [matricesResults_Cascade, sampleResults_Cascade, _, couplingStrengthResults_Cascade, noiseScaleResults_Cascade, _] = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_separate_Cascade, samples=samples_separate,alphas=[],couplingStrengths=couplingStrCascade_separate,noiseScales=noiseScales_separate,tauMaxs=[],
                                            runsPerConfig=runsPerConfigSeparateParams, fakeRun=fakeRun, startSeed=seed, evalType = "Full")
            # save everything regardless of what kind of metric is used later
            matricesResults = np.stack((matricesResults_VAR, matricesResults_Cascade), axis=0)
            sampleResults = np.stack((sampleResults_VAR,sampleResults_Cascade), axis=0)
            couplingStrengthResults = np.stack((couplingStrengthResults_VAR,couplingStrengthResults_Cascade), axis=0)
            noiseScaleResults = np.stack((noiseScaleResults_VAR,noiseScaleResults_Cascade), axis=0)
            if doSave:
                np.savez(folder + dataParamsFileName, matricesResults = matricesResults, sampleResults = sampleResults, couplingStrengthResults = couplingStrengthResults, noiseScaleResults = noiseScaleResults)
        else:
            npzfile = np.load(folder+dataParamsFileName)
            matricesResults = npzfile["matricesResults"]
            sampleResults = npzfile["sampleResults"]
            couplingStrengthResults = npzfile["couplingStrengthResults"] 
            noiseScaleResults = npzfile["noiseScaleResults"]
        # shape: 2 (VAR/Cascade), usually 4 (TP/TN/FP/FN) or previously non-existant axis for F1 scores, 3 (Algorithm), runsPerConfig, iterated values
        matricesResults = MCCFromFull(matricesResults, axis=1)
        sampleResults = MCCFromFull(sampleResults, axis=1)
        couplingStrengthResults = MCCFromFull(couplingStrengthResults, axis=1)
        noiseScaleResults = MCCFromFull(noiseScaleResults, axis=1)
        matricesResults, matricesError = getMeanStdDev(matricesResults, axis = 2)
        sampleResults, sampleError = getMeanStdDev(sampleResults, axis = 2)
        couplingStrengthResults, couplingStrengthError = getMeanStdDev(couplingStrengthResults, axis = 2)
        noiseScaleResults, noiseScaleError = getMeanStdDev(noiseScaleResults, axis = 2)

        #plot samples
        vis.saveMCCCurve(sampleResults[0,:,:], samples_separate, "MCC by sample size - VAR systems", folder + "Samples_VAR", errors=sampleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log", xlabel="Samples", ylabel="MCC")
        vis.saveMCCCurve(sampleResults[1,:,:], samples_separate, "MCC by sample size - x³ systems", folder + "Samples_Cascade", errors=sampleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log", xlabel="Samples", ylabel="MCC")

        #plot couplingStrengths
        vis.saveMCCCurve(couplingStrengthResults[0,:,:], couplingStrVAR_separate, "MCC by coupling strength - VAR systems", folder + "CouplingStrength_VAR",errors=couplingStrengthError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Coupling Strength", ylabel="MCC")
        vis.saveMCCCurve(couplingStrengthResults[1,:,:], couplingStrCascade_separate, "MCC by coupling strength - x³ systems", folder + "CouplingStrength_Cascade", errors=couplingStrengthError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Coupling Strength", ylabel="MCC")

        vis.saveMCCCurve(sampleResults[0,:,:], samples_separate, "MCC by sample size - VAR systems", folder + "Poster_Samples_VAR", xlabel="Number of Samples", ylabel="MCC", figsize=(4.5,3.3), dpi=300, errors=sampleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log")
        vis.saveMCCCurve(sampleResults[1,:,:], samples_separate, "MCC by sample size - x³ systems", folder + "Poster_Samples_Cascade", xlabel="Number of Samples", ylabel="MCC", figsize=(4.5,3.3), dpi=300, errors=sampleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xscale = "log")

        #plot couplingStrengths
        vis.saveMCCCurve(couplingStrengthResults[0,:,:], couplingStrVAR_separate, "MCC by coupling strength - VAR systems", folder + "Poster_CouplingStrength_VAR",xlabel="Coupling Strength",ylabel="MCC",figsize=(4.5,3.3), dpi=300, errors=couplingStrengthError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave)
        vis.saveMCCCurve(couplingStrengthResults[1,:,:], couplingStrCascade_separate, "MCC by coupling strength - x³ systems", folder + "Poster_CouplingStrength_Cascade", xlabel="Coupling Strength",ylabel="MCC",figsize=(4.5,3.3), dpi=300, errors=couplingStrengthError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave)


        #plot noiseScale
        vis.saveMCCCurve(noiseScaleResults[0,:,:], noiseScales_separate, "MCC by noise scale - VAR systems", folder + "Noise_VAR",errors=noiseScaleError[0,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Noise Scale", ylabel="MCC")
        vis.saveMCCCurve(noiseScaleResults[1,:,:], noiseScales_separate, "MCC by noise scale - x³ systems", folder + "Noise_Cascade", errors=noiseScaleError[1,:,:], rowLabels = ["GCSS", "LKIF", "PCMCI"], show=showDiagrams, save = doSave, xlabel="Noise Scale", ylabel="MCC")

        #plot matrices
        matricesResults = np.reshape(matricesResults, (2,3,3,3), order="C")
        matricesError = np.reshape(matricesError, (2,3,3,3), order="C")
        vis.saveMCCCurve(matricesResults[0,0,:,:], [3,6,12], "GCSS - MCC by number of variables \n VAR systems", folder + "GCSS_Matrices_VAR", errors=matricesError[0,0,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveMCCCurve(matricesResults[0,1,:,:], [3,6,12], "LKIF - MCC by number of variables \n VAR systems", folder + "LKIF_Matrices_VAR", errors=matricesError[0,1,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveMCCCurve(matricesResults[0,2,:,:], [3,6,12], "PCMCI - MCC by number of variables \n VAR systems", folder + "PCMCI_Matrices_VAR", errors=matricesError[0,2,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        
        vis.saveMCCCurve(matricesResults[1,0,:,:], [3,6,12], "GCSS - MCC by number of variables \n x³ systems", folder + "GCSS_Matrices_Cascade", errors=matricesError[1,0,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveMCCCurve(matricesResults[1,1,:,:], [3,6,12], "LKIF - MCC by number of variables \n x³ systems", folder + "LKIF_Matrices_Cascade", errors=matricesError[1,1,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")
        vis.saveMCCCurve(matricesResults[1,2,:,:], [3,6,12], "PCMCI - MCC by number of variables \n x³ systems", folder + "PCMCI_Matrices_Cascade", errors=matricesError[1,2,:,:], rowLabels = ["Low Density", "Medium Density", "High Density"], show=showDiagrams, save = doSave, xlabel ="Number of variables", ylabel="MCC")

    if plotDecisionTree:
        if calculateFullSearchSpace:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_combined_VAR, samples=samples_combined,alphas=[],couplingStrengths=couplingStrVAR_combined,noiseScales=noiseScales_combined,tauMaxs=[],
                                            runsPerConfig=runsPerConfigCombinedParams, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = matrices_combined_Cascade, samples=samples_combined,alphas=[],couplingStrengths=couplingStrCascade_combined,noiseScales=noiseScales_combined,tauMaxs=[],
                                            runsPerConfig=runsPerConfigCombinedParams, fakeRun=fakeRun, startSeed=seed, evalType = "Full")
            
            fullResults = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + fullSearchFileName, fullResults)
        else:
            fullResults = np.load(folder+fullSearchFileName)
        fullResultsVAR = fullResults[0]
        fullResultsCascade = fullResults[1]

        #shape: usually 4 (TP/TN/FP/FN) or non-existant dim before, 3 (algorithms), runsPerConfig, len(couplingMatrices), len(samples), 1, len(couplingStrengths), len(noiseScales), 1
        fullResultsVAR = MCCFromFull(fullResultsVAR, axis=0)
        fullResultsCascade = MCCFromFull(fullResultsCascade, axis=0)

        # a data point has the following features: matrixSize, matrixDensity, samples, couplingStrength, noiseScale
        # with the label as some string indicating the ranking of algorithms

        feature_names = ["System Size", "System Density", "Samples", "Coupling Strength", "Noise Scale"]

        dataVAR=[]
        labelsVAR = []
        labelsMaxVAR = []
        for i in range(fullResultsVAR.shape[2]):
            for j in range(fullResultsVAR.shape[3]):
                for k in range(fullResultsVAR.shape[5]):
                    for m in range(fullResultsVAR.shape[6]):
                        for randomRun in range(fullResultsVAR.shape[1]):
                            dataVAR.append([int(i/3), i%3, samples_combined[j], couplingStrVAR_combined[k], noiseScales_combined[m]])
                            labelsVAR.append(getLabel(fullResultsVAR[:,randomRun, i,j,0,k,m,0]))
                            labelsMaxVAR.append(getMaxLabel(fullResultsVAR[:,randomRun, i,j,0,k,m,0]))
        labelsVAR = np.array(labelsVAR)
        labelsMaxVAR = np.array(labelsMaxVAR)
        dataVAR = np.array(dataVAR)
        class_names_max_VAR = ["GCSS", "LKIF", "PCMCI", "Equal"]
        class_names_max_VAR = class_names_max_VAR[np.min(labelsMaxVAR):]
        class_names_VAR = ["PCMCI-GCSS-LKIF", "GCSS-PCMCI-LKIF","GCSS-LKIF-PCMCI","PCMCI-LKIF-GCSS","LKIF-PCMCI-GCSS","LKIF-GCSS-PCMCI"]
        class_names_VAR = class_names_VAR[np.min(labelsVAR):]
        labelsVAR = labelsVAR - np.min(labelsVAR)
        labelsMaxVAR = labelsMaxVAR - np.min(labelsMaxVAR)
        for i in range(6):
            while i not in labelsVAR and i < np.max(labelsVAR):
                labelsVAR = labelsVAR-1
                del class_names_VAR[i]
        for i in range(4):
            while i not in labelsMaxVAR and i < np.max(labelsMaxVAR):
                labelsMaxVAR = labelsMaxVAR-1
                del class_names_max_VAR[i]

        stability = DataGenerator.getStabilityMatrix(matrices_combined_Cascade, samples_combined, couplingStrCascade_combined, noiseScales_combined, 1)
        dataCascade=[]
        labelsCascade = []
        labelsMaxCascade = []
        for i in range(fullResultsCascade.shape[2]):
            for j in range(fullResultsCascade.shape[3]):
                for k in range(fullResultsCascade.shape[5]):
                    for m in range(fullResultsCascade.shape[6]):
                        if stability[i,j,k,m] == 0:
                            for randomRun in range(fullResultsCascade.shape[1]):
                                dataCascade.append([int(i/3), i%3, samples_combined[j], couplingStrCascade_combined[k], noiseScales_combined[m]])
                                labelsCascade.append(getLabel(fullResultsCascade[:,randomRun, i,j,0,k,m,0]))
                                labelsMaxCascade.append(getMaxLabel(fullResultsCascade[:,randomRun, i,j,0,k,m,0]))
        labelsCascade = np.array(labelsCascade)
        labelsMaxCascade = np.array(labelsMaxCascade)
        dataCascade = np.array(dataCascade)
        class_names_max_Cascade = ["GCSS", "LKIF", "PCMCI", "Equal"]
        class_names_max_Cascade = class_names_max_Cascade[np.min(labelsMaxCascade):]
        class_names_Cascade = ["PCMCI-GCSS-LKIF", "GCSS-PCMCI-LKIF","GCSS-LKIF-PCMCI","PCMCI-LKIF-GCSS","LKIF-PCMCI-GCSS","LKIF-GCSS-PCMCI"]
        class_names_Cascade = class_names_Cascade[np.min(labelsCascade):]
        labelsCascade = labelsCascade - np.min(labelsCascade)
        for i in range(6):
            while i not in labelsCascade and i < np.max(labelsCascade):
                labelsCascade = labelsCascade-1
                del class_names_Cascade[i]
        for i in range(4):
            while i not in labelsMaxCascade and i < np.max(labelsMaxCascade):
                labelsMaxCascade = labelsMaxCascade-1
                del class_names_max_Cascade[i]

        vis.saveDecisionTree(dataVAR, labelsMaxVAR, feature_names, class_names_max_VAR, folder + "VAR_DecTree_FirstRank.svg", show=showDiagrams, save= doSave)

        vis.saveDecisionTree(dataCascade, labelsMaxCascade, feature_names, class_names_max_Cascade, folder + "Cascade_DecTree_FirstRank.svg", show=showDiagrams, save= doSave)

    if plotAlphaCombined:
        if calculateAlphaCombined:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=samples_combined,alphas=alphas_combined,couplingStrengths=couplingStrVAR_combined,noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaCombination, fakeRun=fakeRun, startSeed= seed, evalType = "Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=False, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=samples_combined,alphas=alphas_combined,couplingStrengths=couplingStrCascade_combined,noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaCombination, fakeRun=fakeRun, startSeed=seed, evalType = "Full")            
            fullResults = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + alphaCombinedFileName, fullResults)
        else:
            fullResults = np.load(folder+alphaCombinedFileName)
        fullResultsVAR = fullResults[0]
        fullResultsCascade = fullResults[1]

        print(fullResultsCascade[:,1,:,0,3,0,3])
        print(fullResultsCascade[:,2,:,0,3,0,3])

        #shape: usually 4 (TP/TN/FP/FN) or non-existant dim before, 3 (algorithms), runsPerConfig, len(couplingMatrices), len(samples), len(alphas), len(couplingStrengths), len(noiseScales), len(tauMax)
        fullResultsVAR = MCCFromFull(fullResultsVAR, axis=0)
        fullResultsCascade = MCCFromFull(fullResultsCascade, axis=0)

        avgResultsVAR = np.average(fullResultsVAR, axis=1)
        avgResultsCasc = np.average(fullResultsCascade, axis=1)

        bestAlphaVAR = np.array(alphas_combined)[np.argmax(avgResultsVAR, axis=3).astype(int)].squeeze()
        bestAlphaCasc = np.array(alphas_combined)[np.argmax(avgResultsCasc, axis=3).astype(int)].squeeze()

        bestMCCVAR = np.max(avgResultsVAR, axis=3).squeeze()
        bestMCCCasc = np.max(avgResultsCasc, axis=3).squeeze()

        vis.saveHeatmap(bestAlphaVAR[0].T, bestMCCVAR[0].T, "", filename = folder + "GCSS_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        vis.saveHeatmap(bestAlphaVAR[1].T, bestMCCVAR[1].T, "", filename = folder + "LKIF_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        vis.saveHeatmap(bestAlphaVAR[2].T, bestMCCVAR[2].T, "", filename = folder + "PCMCI_alphaComb_VAR", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrVAR_combined)
        
        vis.saveHeatmap(bestAlphaCasc[0].T, bestMCCCasc[0].T, "", filename = folder + "GCSS_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)
        vis.saveHeatmap(bestAlphaCasc[1].T, bestMCCCasc[1].T, "", filename = folder + "LKIF_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)
        vis.saveHeatmap(bestAlphaCasc[2].T, bestMCCCasc[2].T, "", filename = folder + "PCMCI_alphaComb_Cascade", show=showDiagrams, save=doSave, 
                        xlabel="Samples",ylabel="Coupling Strengths", xtickLabels=samples_combined, ytickLabels=couplingStrCascade_combined)

    if plotBonusAlpha:
        if calculateBonusAlpha:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed= seed,iterateSeeds=False, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed=seed,iterateSeeds=False, evalType="Full")
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsAlpha = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + "bonusAlpha.npy", resultsAlpha)
        else:
            resultsAlpha = np.load(folder+"bonusAlpha.npy")
        if calculateBonusAlpha2:
            seed, results_VAR = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "VAR", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixVAR, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthVAR, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed= seed,iterateSeeds=True, evalType="Full")
            seed, results_Cascade = runAccuracyEvaluation(algorithms=["GCSS", "LKIF", "PCMCI"], model= "Cascade", evaluateSeparately=True, 
                                            defaultCouplingMatrix= defaultCouplingMatrixCascade, defaultSamples=defaultSamples, defaultAlpha= 0.05, 
                                            defaultCouplingStrength= defaultCouplingStrengthCascade, defaultNoiseScale= defaultNoiseScale,  defaultTauMax=defaultTauMax, 
                                            couplingMatrices = [], samples=[],alphas=alphas,couplingStrengths=[],noiseScales=[],tauMaxs=[],
                                            runsPerConfig=runsPerConfigAlphaBonus, fakeRun=fakeRun, startSeed=seed,iterateSeeds=True, evalType="Full")
            results_VAR = results_VAR[2]
            results_Cascade = results_Cascade[2]
            resultsAlpha2 = np.stack((results_VAR, results_Cascade), axis=0)
            if doSave:
                np.save(folder + "bonusAlpha2.npy", resultsAlpha2)
        else:
            resultsAlpha2 = np.load(folder + "bonusAlpha2.npy")

        print(np.squeeze(resultsAlpha[0,:,2,:,10]))
        print(np.squeeze(resultsAlpha2[0,:,2,:,10]))

        resultsAlpha = np.average(tpr_fpr_FromFull(resultsAlpha, axis= 1), axis= 3)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsAlpha[:,0,0,:], resultsAlpha[:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha[:,0,1,:], resultsAlpha[:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha[:,0,2,:], resultsAlpha[:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade_bonus", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        
        resultsAlpha2 = np.average(tpr_fpr_FromFull(resultsAlpha2, axis= 1), axis= 3)
        # visualization for default configuration with 3 vertices, 4 edges
        vis.saveROCCurve(resultsAlpha2[:,0,0,:], resultsAlpha2[:,1,0,:], alphas, "GCSS ROC Curve for alpha values", folder+"GCSS_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha2[:,0,1,:], resultsAlpha2[:,1,1,:], alphas, "LKIF ROC Curve for alpha values", folder+"LKIF_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")
        vis.saveROCCurve(resultsAlpha2[:,0,2,:], resultsAlpha2[:,1,2,:], alphas, "PCMCI ROC Curve for alpha values", folder+"PCMCI_Alpha_VAR_Cascade", rowLabels=["VAR", "x³"], show=showDiagrams, save=doSave, dpi=200, xlabel="False Positive Rate", ylabel ="True Positive Rate")


    endTime = time.process_time()
    print("Total execution time in seconds: ")
    print(endTime - startTime)
    print(str(int((endTime - startTime)/60))+ " minutes")
    
# def main():
#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     plotOnly = True
#     randomRuns = 2
#     alpha = 0.05
#     samples = 1000
#     couplingStrength = 1
#     tauMax = [5,1,5]

#     if rank == 0:
#         sample6dEvaluations(plotOnly = plotOnly,
#         sampleCounts = [50, 100, 200, 500, 1000, 2000, 5000, 10000],
#         alpha = alpha,
#         randomRuns = randomRuns,
#         tauMax = tauMax,
#         couplingStrength=couplingStrength)
    
#     if rank == 1:
#         couplStrength6dEvaluations(plotOnly = plotOnly,
#         couplStrengths = np.array([0.01, 0.02, 0.05, 0.07,0.1,0.15,0.2,0.25,0.3]),
#         samples = samples,
#         alpha = alpha,
#         randomRuns = randomRuns,
#         tauMax = tauMax)

#     if rank == 2:
#         delay6dEvaluations(plotOnly = plotOnly,
#         delaySizes = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0, 1.5,2.0,3.0],
#         samples = samples,
#         alpha = alpha,
#         randomRuns = randomRuns,
#         couplStrength=couplingStrength)

#     if rank == 3:
#         system6dEvaluations(plotOnly = plotOnly,
#         alpha = alpha,
#         samples = samples,
#         randomRuns = randomRuns,
#         tauMax = tauMax,
#         couplingStrength=couplingStrength)

#     if rank == 4:
#         nonStationaryStable(plotOnly = plotOnly,
#         ceilings = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0],
#         alpha = alpha,
#         samples = samples,
#         tauMax = tauMax,
#         randomRuns = randomRuns)

if __name__ == "__main__":
    nonStationaryStable(plotOnly = True,
        ceilings = [0, 0.1, 0.5, 0.8],
        alpha = 0.05,
        samples = 1000,
        tauMax = [5,1,5],
        randomRuns = 3)