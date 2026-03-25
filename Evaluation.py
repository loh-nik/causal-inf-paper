from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Some of these libraries use jit compilation on import, we avoid race conditions by importing once in the main rank
if rank == 0:
    print("Rank 0 importing PCMCI first...")
    import Visualization as vis
    import DataGenerator
    import PCMCI
    import numpy as np
    import time
    from itertools import product
    import GCSS
    import LKIF
    import os
    comm.barrier()  # let others wait

else:
    comm.barrier()
    import PCMCI
    import numpy as np
    import DataGenerator
    import Visualization as vis
    import time
    from itertools import product
    import GCSS
    import LKIF


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

### HELPER METHODS FOR METRICS

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
    """Calculates the Matthews Correlation Coefficient for a list of absolute numbers of true positives, TN, FP, FN.

    Args:
        arr (list): list of absolute numbers of true positives, TN, FP, FN.

    Returns:
        float: MCC between -1 (everything predicted falsely) to 1 (everything predicted correctly)
    """
    truePos, trueNeg, falsePos, falseNeg = arr
    return float(((truePos * trueNeg) - (falsePos * falseNeg)) / np.sqrt((truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg)) if (truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg + falseNeg) != 0 else 0)

def MCCFromFull(a, axis=0):
    """Helper method to calculate Matthews Correlation Coefficient along axes

    Args:
        a (nparray): Array which contains TP, TN, FP, FN along some axis.
        axis (int, optional): axis along which to calculate MCC. Defaults to 0.

    Returns:
        nparray: resulting array with float entries indicating MCC.
    """
    if a.shape[int(axis)] != 4:
        print("Error: Can't determine MCC from falsely shaped array")
    return np.apply_along_axis(MCC, axis, a)

def tpr_fpr_Scores(a,b):
    """Return true positive and false positive rates given some ground truth and prediction.

    Args:
        a (nparray(n x n)): Ground truth matrix
        b (nparray(n x n)): Predicted matrix

    Returns:
        list(float): Pair containing [tpr, fpr]
    """
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

# HELPER METHODS FOR GRAPH GENERATION

def generate_random_graphs():
    import networkx as nx
    G = nx.from_numpy_array(mediumCouplingMatrixCascade_HighDense, create_using=nx.DiGraph)
    cycles = list(nx.simple_cycles(G))
    originalCycles = len(cycles)
    originalEdges = len(G.edges)
    print("Original graph cycles:", originalCycles)
    print("Original graph edges:", originalEdges)

    finalGraphs = [mediumCouplingMatrixCascade_HighDense]
    seed = 0
    np.random.seed(42)
    # generate 10 graphs beyond the first one
    while len(finalGraphs) < 11:
        print("Graph attempt")
        G = nx.gnm_random_graph(6, originalEdges, directed=True, seed= seed)
        seed += 1
        cycles = list(nx.simple_cycles(G))
        if len(cycles) == originalCycles:
            matrix = nx.to_numpy_array(G)
            edges = np.argwhere(matrix == 1)
            neg_idx = np.random.choice(len(edges), size=int(originalEdges/2), replace=False)
            for i, j in edges[neg_idx]:
                matrix[i, j] = -1
            finalGraphs.append(matrix)
    np.save("data/random_graphs.npy", np.array(finalGraphs))
    return finalGraphs

def get_random_graphs():
    graphs = np.load("data/random_graphs.npy")
    return graphs

# EVALUATION

def getMetricOfRealization(couplingMatrix, algorithms, model, samples, alpha, couplingStrength, noiseScale, tauMax, seed, evalType, delayLength = 0, fullData = [], returnMatrices = False, verbose =False, deltaT = 0.1):
    """Generates data given some coupling matrix, applies the causal methods and evaluates them according to the chosen metric.

    Args:
        couplingMatrix (_nparray_): n x n matrix which contains 1 for a coupling between variables and 0 for non-coupled variables
        algorithms (_list(str)_): List of causal methods applied to the data. Valid entries: PCMCI, LKIF, GCSS
        model (_str_): The data generation model, either "VAR" for a vector autoregressive model or "Cascade" for a network of interacting conceptual tipping elements.
        samples (_int_): Number of generated samples
        alpha (_float_): Error rate, in terms of false positives, provides a significance approximation
        couplingStrength (_float_): Multiplies all non-diagonal coefficients of the coupling matrix by this value
        noiseScale (_float_): Scaling of the white noise on all variables
        tauMax (_int_): Max. time steps across which analysis should be conducted
        seed (_int_): Seed for white noise generation, only implemented for VAR model
        evalType (_str_): Collected evaluation metrics, either "tpr_fpr" or "Full" for TP, FP, TN, FN
        delayLength (int, optional): Optional delay for all interactions, given in number of samples. Defaults to 0.
        fullData (list, optional): If neither data model is chosen, this data is provided to the tests instead, for manual bypassing. Defaults to [].
        returnMatrices (bool, optional): If true, bypass the metrics step and immediately return the detected matrices of the causal methods. Defaults to False.
        verbose (bool, optional): If true, the data generating process may display a progress bar. Defaults to False.
        deltaT (float, optional): The timestep between samples of the ODE

    Returns:
        nparray: By default, returns an nparray with the metrics in the shape (len(algorithms), metric_length), where metric_lenght is 2 if evalType is "tpr_fpr", and 4 if evalType is "Full". If returnMatrices is True, instead return prediction matrices in shape (len(algorithms), n, n).
    """
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
        fullData = np.array(DataGenerator.getCascadeDataBrainpy(matrix, samples, noiseScale, delayLength, verbose, deltaT)).T
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
    if "PCMCI_GPDC" in algorithms:
        tauPCMCI = tauMax
        if tauList: 
            tauPCMCI = tauMax[algorithms.index("PCMCI")]
        matrixPCMCI, p_values = PCMCI.PCMCIPlus(fullData, [], range(fullData.shape[1]), None, tauPCMCI, alpha if alpha<=1 else 1, contempLinks=True, independenceTest="GPDC")
        graph_bool = p_values <= alpha
        matrixPCMCI = maxSignificantLink(matrixPCMCI, graph_bool, axis = 2)
        np.fill_diagonal(matrixPCMCI,0)
        matrices.append(matrixPCMCI.T)
    if "PCMCI_robustpc" in algorithms:
        tauPCMCI = tauMax
        if tauList: 
            tauPCMCI = tauMax[algorithms.index("PCMCI")]
        matrixPCMCI, p_values = PCMCI.PCMCIPlus(fullData, [], range(fullData.shape[1]), None, tauPCMCI, alpha if alpha<=1 else 1, contempLinks=True, independenceTest="robustparcorr")
        graph_bool = p_values <= alpha
        matrixPCMCI = maxSignificantLink(matrixPCMCI, graph_bool, axis = 2)
        np.fill_diagonal(matrixPCMCI,0)
        matrices.append(matrixPCMCI.T)
    np.fill_diagonal(couplingMatrix, 0)
    if returnMatrices:
        return np.array(matrices)
    if evalType == "tpr_fpr":
        metrics = np.array([tpr_fpr_Scores(couplingMatrix, matrix) for matrix in matrices])
    elif evalType == "Full":
        metrics = np.array([getFullMetrics(couplingMatrix, matrix) for matrix in matrices])
    else:
        print("Error: Invalid evaluation type found")
    return metrics.T

# EXPERIMENTS

# experiment on delay length
def delay6dEvaluations(plotOnly, delaySizes, samples, alpha, randomRuns, couplStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        # this data should have: 3 variables, 100 runs, 2000 samples per run
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(delaySizes), randomRuns, 4, 3))

        seed = comm.Get_rank()
        truthMatrixVAR = mediumCouplingMatrixVAR_LowDense
        fullOutVAR = np.zeros((len(delaySizes), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(delaySizes)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        localFullVAR = []
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = samples, alpha= alpha, couplingStrength= couplStrength, noiseScale= 0.01, tauMax= int(delaySizes[j] * 10) + 1,
                                                seed= 0, evalType= "Full", delayLength=delaySizes[j], verbose=verbose)
            localFull.append((j, i, metrics))

            metrics = getMetricOfRealization(couplingMatrix = truthMatrixVAR, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                samples = samples, alpha= alpha, couplingStrength= couplStrength * 0.1, noiseScale= 0.01, tauMax= int(delaySizes[j] * 10) + 1,
                                                seed= seed, evalType= "Full", delayLength=int(delaySizes[j] * 10), verbose=verbose)
            seed += comm.Get_size()
            localFullVAR.append((j,i,metrics))

        gathered = comm.gather(localFull, root=0)
        gatheredVAR = comm.gather(localFullVAR, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
            for result in gatheredVAR:
                for j, i, value in result:
                    fullOutVAR[j, i] = value
            np.save(data_dir + "/6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
            np.save(data_dir + "/VAR_6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy", fullOutVAR)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy")
        fullOutVAR = np.load(data_dir + "/VAR_6d_multiDelay_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, delaySizes, "", diag_dir + "/delays6dCascades_Paper", stdDev.T, greyAxisAt=0, show=False, save = True, rowLabels = [], xlabel = "Delay (no unit)", ylabel ="",yTickLabels=False,yLims=[-0.23, 1.03], fontsizeFactor=1.2)
        vis.saveMCCCurve(mean.T, delaySizes, "", diag_dir + "/delays6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay (no unit)", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(median.T, delaySizes, "", diag_dir + "/delays6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay (no unit)", ylabel ="Matthews Correlation Coefficient")

        # same for VAR systems
        scores = MCCFromFull(fullOutVAR, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, [int(d*10) for d in delaySizes], "", diag_dir + "/delays6dVAR", stdDev.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay in time steps", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(median.T, [int(d*10) for d in delaySizes], "", diag_dir + "/delays6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Delay in time steps", ylabel ="Matthews Correlation Coefficient")

        scores = tpr_fpr_FromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis=1)
        print(mean.shape)

        mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)
        stdDev = np.append(stdDev[:,0,:], stdDev[:,1,:], axis=1)
        print(mean.shape)

        vis.saveMCCCurve(mean.T, delaySizes, "", diag_dir + "/delays6dCascades_TPR", stdDev.T, show=False, save = True, rowLabels=["GCSS-TPR", "LKIF-TPR", "PCMCI-TPR", "GCSS-FPR", "LKIF-FPR", "PCMCI-FPR"], xlabel = "Delay (no unit)", ylabel ="Rate")

# main sample size experiment
def sampleEvaluations(plotOnly, sampleCounts, alpha, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(sampleCounts), randomRuns, 4, 3))
        seed = comm.Get_rank()
        truthMatrixVAR = mediumCouplingMatrixVAR_LowDense
        fullOutVAR = np.zeros((len(sampleCounts), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(sampleCounts)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        localFullVAR = []
        print("Node " + str(comm.Get_rank()) + " executing " + str(len(param_combs)) + " combinations for sample test")
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, metrics))

            metrics = getMetricOfRealization(couplingMatrix = truthMatrixVAR, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength*0.1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= seed, evalType= "Full", verbose=verbose)
            seed += comm.Get_size()
            localFullVAR.append((j, i, metrics))
        print("Node " + str(comm.Get_rank()) + " finished sample test")
        gathered = comm.gather(localFull, root=0)
        gatheredVAR = comm.gather(localFullVAR, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
            for result in gatheredVAR:
                for j, i, value in result:
                    fullOutVAR[j, i] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_Samples_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
            np.save(data_dir + "/VAR_6d_Samples_"+str(randomRuns)+"_runs_metrics.npy", fullOutVAR)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_Samples_"+str(randomRuns)+"_runs_metrics.npy")
        fullOutVAR = np.load(data_dir + "/VAR_6d_Samples_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/samples6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/samples6dCascades_Paper", stdDev.T, greyAxisAt=1000, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient", xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, figsize=(5.5,4))
        vis.saveMCCCurve(median.T, sampleCounts, "", diag_dir + "/samples6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient",xscale ="log")
        
        scores = MCCFromFull(fullOutVAR, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/samples6dVAR", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        vis.saveMCCCurve(median.T, sampleCounts, "", diag_dir + "/samples6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        
        scores = tpr_fpr_FromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis=1)

        mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)
        stdDev = np.append(stdDev[:,0,:], stdDev[:,1,:], axis=1)

        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/samples6dCascades_TPR", stdDev.T, show=False, save = True, rowLabels=["GCSS-TPR", "LKIF-TPR", "PCMCI-TPR", "GCSS-FPR", "LKIF-FPR", "PCMCI-FPR"], xlabel = "Delay (no unit)", ylabel ="Rate", xscale ="log")

# appendix experiment for different conditional independence tests of PCMCI
def sampleEvaluationsAppendix(plotOnly, sampleCounts, alpha, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(sampleCounts), randomRuns, 4, 3))
        seed = comm.Get_rank()
        param_combs = list(product(np.arange(len(sampleCounts)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        print("Node " + str(comm.Get_rank()) + " executing " + str(len(param_combs)) + " combinations for appendix sample test")
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["PCMCI", "PCMCI_robustpc", "PCMCI_GPDC"], model= "Cascade", 
                                                samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, metrics))

            seed += comm.Get_size()
        print("Node " + str(comm.Get_rank()) + " finished appendix sample test")
        gathered = comm.gather(localFull, root=0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_AppendixSamples_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_AppendixSamples_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/Appendix_samples6dCascades", stdDev.T, show=False, save = True, rowLabels=["ParCorr", "Robust-ParCorr", "GPDC"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        vis.saveMCCCurve(median.T, sampleCounts, "", diag_dir + "/Appendix_samples6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=False, save = True, rowLabels=["ParCorr", "Robust-ParCorr", "GPDC"], xlabel = "Number of Samples", ylabel ="Matthews Correlation Coefficient",xscale ="log")
        
        scores = tpr_fpr_FromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis=1)
        print(mean.shape)

        mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)
        stdDev = np.append(stdDev[:,0,:], stdDev[:,1,:], axis=1)
        print(mean.shape)

        vis.saveMCCCurve(mean.T, sampleCounts, "", diag_dir + "/Appendix_samples6dCascades_TPR", stdDev.T, show=False, save = True, rowLabels=["PCMCI-TPR", "PCMCI_robustpc-TPR", "PCMCI_GPDC-TPR", "PCMCI-FPR", "PCMCI_robustpc-FPR", "PCMCI_GPDC-FPR"], xlabel = "Number of Samples", ylabel ="Rate", xscale ="log")

# experiment on coupling strength
def couplStrength6dEvaluations(plotOnly, couplStrengths, samples, alpha, randomRuns, tauMax, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(couplStrengths), randomRuns, 4, 3))
        seed = comm.Get_rank()
        truthMatrixVAR = mediumCouplingMatrixVAR_LowDense
        fullOutVAR = np.zeros((len(couplStrengths), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(couplStrengths)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        localFullVAR = []
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = samples, alpha= alpha, couplingStrength=  couplStrengths[j] * 10, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, metrics))
            metrics = getMetricOfRealization(couplingMatrix = truthMatrixVAR, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                samples = samples, alpha= alpha, couplingStrength= couplStrengths[j], noiseScale= 0.01, tauMax= tauMax,
                                                seed= seed, evalType= "Full", verbose=verbose)
            seed += comm.Get_size()
            localFullVAR.append((j, i, metrics))
        gathered = comm.gather(localFull, root=0)
        gatheredVAR = comm.gather(localFullVAR, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
            for result in gatheredVAR:
                for j, i, value in result:
                    fullOutVAR[j, i] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
            np.save(data_dir + "/VAR_6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy", fullOutVAR)
    elif comm.Get_rank() == 0: 
        fullOut = np.load(data_dir + "/6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy")
        fullOutVAR = np.load(data_dir + "/VAR_6d_CouplStren_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 90% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, couplStrengths*10, "", diag_dir + "/coupl6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(mean.T, couplStrengths*10, "", diag_dir + "/coupl6dCascades_Paper", stdDev.T, greyAxisAt=1, show=False, save = True, rowLabels=[], xlabel = "Coupling Strength", ylabel ="", yTickLabels=False, yLims=[-0.23, 1.03], fontsizeFactor=1.2)
        vis.saveMCCCurve(median.T, couplStrengths*10, "", diag_dir + "/coupl6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="Matthews Correlation Coefficient")

        scores = MCCFromFull(fullOutVAR, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 90% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, couplStrengths, "", diag_dir + "/coupl6dVAR", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(median.T, couplStrengths, "", diag_dir + "/coupl6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Coupling Strength", ylabel ="Matthews Correlation Coefficient")

        scores = tpr_fpr_FromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis=1)
        print(mean.shape)

        mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)
        stdDev = np.append(stdDev[:,0,:], stdDev[:,1,:], axis=1)
        print(mean.shape)

        vis.saveMCCCurve(mean.T, couplStrengths, "", diag_dir + "/coupl6dCascades_TPR", stdDev.T, show=False, save = True, rowLabels=["GCSS-TPR", "LKIF-TPR", "PCMCI-TPR", "GCSS-FPR", "LKIF-FPR", "PCMCI-FPR"], xlabel = "Delay (no unit)", ylabel ="Rate")

def scale_lightness(rgb, scale_l):
    import colorsys
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def subtractedMatrix(matrix, desiredDim):
    result = np.zeros((desiredDim, desiredDim))
    for i in range(desiredDim):
        result[i] = matrix[i,:desiredDim]
    return result

# conduct the experiment with different system sizes and densities
def system6dEvaluations(plotOnly, alpha, samples, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    import matplotlib as mpl
    if not comm:
        comm = MPI.COMM_WORLD
    defaultCols = mpl.color_sequences["tab10"]
    # low density: n-1 edges
    # high density: 2n (or 2n-1 for 3 nodes)
    sizes = [3,4,5,6,7,8,9,10,11,12]
    sizeLen = len(sizes)
    cascMatrices = [
        defaultCouplingMatrixCascade_LowDense,
                    subtractedMatrix(mediumCouplingMatrixCascade_LowDense, 4),subtractedMatrix(mediumCouplingMatrixCascade_LowDense, 5), mediumCouplingMatrixCascade_LowDense,
                     subtractedMatrix(largeCouplingMatrixCascade_LowDense, 7),subtractedMatrix(largeCouplingMatrixCascade_LowDense, 8),subtractedMatrix(largeCouplingMatrixCascade_LowDense, 9),subtractedMatrix(largeCouplingMatrixCascade_LowDense, 10),
                      subtractedMatrix(largeCouplingMatrixCascade_LowDense, 11), largeCouplingMatrixCascade_LowDense, 
                      defaultCouplingMatrixCascade_HighDense, 
                      subtractedMatrix(mediumCouplingMatrixCascade_HighDense, 4),subtractedMatrix(mediumCouplingMatrixCascade_HighDense, 5), mediumCouplingMatrixCascade_HighDense,
                     subtractedMatrix(largeCouplingMatrixCascade_HighDense, 7),subtractedMatrix(largeCouplingMatrixCascade_HighDense, 8),subtractedMatrix(largeCouplingMatrixCascade_HighDense, 9),subtractedMatrix(largeCouplingMatrixCascade_HighDense, 10),
                      subtractedMatrix(largeCouplingMatrixCascade_HighDense, 11), largeCouplingMatrixCascade_HighDense
                      ]
    VARMatrices = [
        defaultCouplingMatrixVAR_LowDense,
                    subtractedMatrix(mediumCouplingMatrixVAR_LowDense, 4),subtractedMatrix(mediumCouplingMatrixVAR_LowDense, 5), mediumCouplingMatrixVAR_LowDense,
                     subtractedMatrix(largeCouplingMatrixVAR_LowDense, 7),subtractedMatrix(largeCouplingMatrixVAR_LowDense, 8),subtractedMatrix(largeCouplingMatrixVAR_LowDense, 9),subtractedMatrix(largeCouplingMatrixVAR_LowDense, 10),
                      subtractedMatrix(largeCouplingMatrixVAR_LowDense, 11), largeCouplingMatrixVAR_LowDense, 
                      defaultCouplingMatrixVAR_HighDense, 
                      subtractedMatrix(mediumCouplingMatrixVAR_HighDense, 4),subtractedMatrix(mediumCouplingMatrixVAR_HighDense, 5), mediumCouplingMatrixVAR_HighDense,
                     subtractedMatrix(largeCouplingMatrixVAR_HighDense, 7),subtractedMatrix(largeCouplingMatrixVAR_HighDense, 8),subtractedMatrix(largeCouplingMatrixVAR_HighDense, 9),subtractedMatrix(largeCouplingMatrixVAR_HighDense, 10),
                      subtractedMatrix(largeCouplingMatrixVAR_HighDense, 11), largeCouplingMatrixVAR_HighDense
                      ]
    if len(cascMatrices) != len(VARMatrices):
        raise ValueError("Matrices for cubic systems and VAR systems not identical length")
    
    if not plotOnly:
        fullOut = np.zeros((len(cascMatrices), randomRuns, 4, 3))
        seed = comm.Get_rank()
        fullOutVAR = np.zeros((len(cascMatrices), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(cascMatrices)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        localFullVAR = []
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = cascMatrices[j], algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                            samples = samples, alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                            seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, metrics))
            metrics = getMetricOfRealization(couplingMatrix = VARMatrices[j], algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplingStrength*0.1, noiseScale= 0.01, tauMax= tauMax,
                                                  seed= seed, evalType= "Full", verbose=verbose)
            seed += comm.Get_size()
            localFullVAR.append((j, i, metrics))
        gathered = comm.gather(localFull, root=0)
        gatheredVAR = comm.gather(localFullVAR, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
            for result in gatheredVAR:
                for j, i, value in result:
                    fullOutVAR[j, i] = value
        if not comm or comm.Get_rank() == 0:
            np.save(data_dir + "/6d_System_ext_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
            np.save(data_dir + "/VAR_6d_System_ext_"+str(randomRuns)+"_runs_metrics.npy", fullOutVAR)
    elif not comm or comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_System_ext_"+str(randomRuns)+"_runs_metrics.npy")
        fullOutVAR = np.load(data_dir + "/VAR_6d_System_ext_"+str(randomRuns)+"_runs_metrics.npy")
    if not comm or comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        lowDense = mean[:sizeLen]
        highDense = mean[sizeLen:]
        lowDenseErr = stdDev[:sizeLen]
        highDenseErr = stdDev[sizeLen:]
        final = np.append(lowDense, highDense, axis=1)
        finalErr = np.append(lowDenseErr, highDenseErr, axis=1)

        vis.saveMCCCurve(final[:,::3].T, sizes, "", diag_dir + "/SystemCascGCSS", [], show=False, save = True, rowLabels=["Low Density", "High Density"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(final[:,1::3].T, sizes, "", diag_dir + "/SystemCascLKIF", [], show=False, save = True, rowLabels=["LKIF-LD", "LKIF-HD"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(final[:,2::3].T, sizes, "", diag_dir + "/SystemCascPCMCI", [], show=False, save = True, rowLabels=["PCMCI-LD","PCMCI-HD"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        
        vis.saveMCCCurveSubplot(3, 1, np.array([final[:,2::3].T, final[:,1::3].T, final[:,::3].T]), sizes, "", diag_dir + "/SystemCascSubplots",
                                 errors = np.array([finalErr[:,2::3].T, finalErr[:,1::3].T, finalErr[:,::3].T]), show=False, save = True, rowLabels=[], figsize=(4,10),
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        
        vis.saveMCCCurve(final.T, sizes, "", diag_dir + "/SystemCasc", [], show=False, save = True, rowLabels=["GCSS-LD", "LKIF-LD", "PCMCI-LD","GCSS-HD", "LKIF-HD", "PCMCI-HD"], 
                        colors=[defaultCols[0], defaultCols[1], defaultCols[2], scale_lightness(defaultCols[0], 1.6), scale_lightness(defaultCols[1], 1.6), scale_lightness(defaultCols[2], 1.6)], xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")

        scores = tpr_fpr_FromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis=1)
        lowDense = mean[:sizeLen]
        highDense = mean[sizeLen:]
        mean = np.append(lowDense, highDense, axis=2)

        mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)

        vis.saveMCCCurve(mean[:,::3].T, sizes, "", diag_dir + "/SystemCascGCSS-TPR", [], show=False, save = True, rowLabels=["GCSS-LD-TPR", "GCSS-HD-TPR","GCSS-LD-FPR", "GCSS-HD-FPR"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(mean[:,1::3].T, sizes, "", diag_dir + "/SystemCascLKIF-TPR", [], show=False, save = True, rowLabels=["LKIF-LD-TPR", "LKIF-HD-TPR","LKIF-LD-FPR", "LKIF-HD-FPR"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(mean[:,2::3].T, sizes, "", diag_dir + "/SystemCascPCMCI-TPR", [], show=False, save = True, rowLabels=["PCMCI-LD-TPR", "PCMCI-HD-TPR","PCMCI-LD-FPR", "PCMCI-HD-FPR"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")

        scores = MCCFromFull(fullOutVAR, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        lowDense = mean[:sizeLen]
        highDense = mean[sizeLen:]
        final = np.append(lowDense, highDense, axis=1)

        vis.saveMCCCurve(final.T, sizes, "", diag_dir + "/SystemVAR", [], show=False, save = True, rowLabels=["GCSS-LD", "LKIF-LD", "PCMCI-LD","GCSS-HD", "LKIF-HD", "PCMCI-HD"],
                        colors=[defaultCols[0], defaultCols[1], defaultCols[2], scale_lightness(defaultCols[0], 1.6), scale_lightness(defaultCols[1], 1.6), scale_lightness(defaultCols[2], 1.6)], xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(final[:,::3].T, sizes, "", diag_dir + "/SystemVARGCSS", [], show=False, save = True, rowLabels=["GCSS-LD", "GCSS-HD"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(final[:,1::3].T, sizes, "", diag_dir + "/SystemVARLKIF", [], show=False, save = True, rowLabels=["LKIF-LD", "LKIF-HD"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")
        vis.saveMCCCurve(final[:,2::3].T, sizes, "", diag_dir + "/SystemVARPCMCI", [], show=False, save = True, rowLabels=["PCMCI-LD","PCMCI-HD"], 
                        xlabel = "Variable Count", ylabel ="Matthews Correlation Coefficient")

# conduct the forcing / nonstationarity experiment
def nonStationaryExp(plotOnly, ceilings, alpha, samples, tauMax, randomRuns, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    """We increase forcing linearly over the course of 50 units of time (500 samples), up to some ceiling, then compare performances across ceiling heights.
    Experiment only conducted for nonlinear system, as there's no tipping in the VAR system"""
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        dim = 6
        metricsNormal = np.zeros((len(ceilings), randomRuns, 4, 3))
        metricsReducedInfo = np.zeros((len(ceilings), randomRuns, 4, 3))
        tipped = np.zeros(len(ceilings))
        param_combs = list(product(np.arange(len(ceilings)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFullInfo = []
        localReducedInfo = []
        localTipped = []
        for i,j in param_combs:
            dT = lambda x,t : (t <= 50) * (-0.4 / 50) * ceilings[i]
            data = DataGenerator.getCascade6dConfoundedBrainpy(dT, truthMatrix, samples, 0.001, verbose=verbose)
            # if we cross the threshold between +1 and -1, we assume tipped
            if np.min(data[:dim]) < 0: localTipped.append((i,j,1))
            matrices = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "None", 
                                                samples = samples, alpha= alpha, couplingStrength= 1, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", fullData = data.T, returnMatrices=True, verbose=verbose)
            matrices = matrices[:,:dim,:dim]
            localFullInfo.append((i,j, np.array([getFullMetrics(truthMatrix, matrix) for matrix in matrices]).T))
            matrices2 = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "None", 
                                                samples = samples, alpha= alpha, couplingStrength= 1, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", fullData = data[:dim].T, returnMatrices=True, verbose=verbose)
            localReducedInfo.append((i,j, np.array([getFullMetrics(truthMatrix, matrix) for matrix in matrices2]).T))
        gathered = comm.gather(localFullInfo, root=0)
        gatheredRed = comm.gather(localReducedInfo, root = 0)
        gatheredTipped = comm.gather(localTipped, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    metricsNormal[j, i] = value
            for result in gatheredRed:
                for j, i, value in result:
                    metricsReducedInfo[j, i] = value
            for result in gatheredTipped:
                for i,j,value in result:
                    tipped[i] += value
            tipped = tipped / randomRuns
            np.save(data_dir + "/Casc_forcing6d.npy", metricsNormal)
            np.save(data_dir + "/Casc_forcing_reducedInfo_6d.npy", metricsReducedInfo)
            np.save(data_dir + "/Casc_forcing6d_tippedFraction.npy", tipped)
    elif comm.Get_rank() == 0:
        metricsNormal = np.load(data_dir + "/Casc_forcing6d.npy")
        metricsReducedInfo = np.load(data_dir + "/Casc_forcing_reducedInfo_6d.npy")
        tipped = np.load(data_dir + "/Casc_forcing6d_tippedFraction.npy")
    if comm.Get_rank() == 0:
        vis.saveMCCCurve(tipped, ceilings, "", diag_dir + "/forcing6d_tippedFraction", xlabel="Forcing Strength", ylabel="", figsize = (10,2),dpi=300, fontsizeFactor=1.4)

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
        
        vis.saveMCCCurve(fullInfoMean.T, ceilings, "", diag_dir + "/forcing6d", fullInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
        vis.saveMCCCurve(noInfoMean.T, ceilings, "", diag_dir + "/forcing6d_reducedInformation", noInfoStd.T, rowLabels=["GCSS", "LKIF", "PCMCI"])
        
        vis.saveMCCScatter(gcssInfo.T, ceilings, "", diag_dir + "/forcing6d_gcss", gcssStd.T, rowLabels=["Confounder Inclusion", "Confounder Exclusion"],figsize=(8,2), dpi=300, legend=False)
        vis.saveMCCScatter(lkifInfo.T, ceilings, "", diag_dir + "/forcing6d_lkif", lkifStd.T, rowLabels=["Confounder Inclusion", "Confounder Exclusion"],figsize=(8,2), dpi=300, legend=False)
        vis.saveMCCScatter(pcmInfo.T, ceilings, "", diag_dir + "/forcing6d_pcmci", pcmStd.T, rowLabels=["Confounder Inclusion", "Confounder Exclusion"],figsize=(8,2), dpi=300, legend=False)
        
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

        import matplotlib as mpl
        defaultCols = mpl.color_sequences["tab10"]
        flippedCols = [defaultCols[0], defaultCols[0], defaultCols[1], defaultCols[1]]
        vis.saveMCCScatter(gcssInfo.T, ceilings, "", diag_dir + "/forcing6d_gcss_TPR", gcssStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR", "Hidden Confounder TPR", "Hidden Confounder FPR"],
                           colors = flippedCols, linestyles=["solid", "dotted", "solid", "dotted"], figsize=(10,2), dpi=300, legend_outside=True, ylim = (-0.05,1.05), legend=False, xTickLabels=False, fontsizeFactor=1.4)
        vis.saveMCCScatter(lkifInfo.T, ceilings, "", diag_dir + "/forcing6d_lkif_TPR", lkifStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR","Hidden Confounder TPR",  "Hidden Confounder FPR"],
                           colors = flippedCols, linestyles=["solid", "dotted", "solid", "dotted"], figsize=(10,2), dpi=300, legend_outside=True, ylim = (-0.05,1.05), legend=False, xTickLabels=False, fontsizeFactor=1.4)
        vis.saveMCCScatter(pcmInfo.T, ceilings, "", diag_dir + "/forcing6d_pcmci_TPR", pcmStd.T, rowLabels=["Known Confounder TPR", "Known Confounder FPR","Hidden Confounder TPR",  "Hidden Confounder FPR"],
                           colors = flippedCols, linestyles=["solid", "dotted", "solid", "dotted"], figsize=(10,2), dpi=300, legend_outside=True, ylim = (-0.05,1.05), legend=False, xTickLabels=False, fontsizeFactor=1.4)
        vis.saveMCCScatter(pcmInfo.T, ceilings, "", diag_dir + "/forcing6d_pcmci_TPR_withLegend", pcmStd.T, rowLabels=["Confounder Inclusion TPR", "Confounder Inclusion FPR","Confounder Exclusion TPR",  "Confounder Exclusion FPR"],
                           colors = flippedCols, linestyles=["solid", "dotted", "solid", "dotted"], figsize=(15,2), dpi=300, legend_outside=True, ylim = (-0.05,1.05), legend=True, xTickLabels=False, fontsizeFactor=1.4, legendColumns=2)

# conduct the runtime experiment with all three methods
def runtimeEvaluations(plotOnly, matrix, alpha, samples, tauMax, randomRuns, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        return
    if not plotOnly:
        runtimes = np.zeros((randomRuns, 3))
        data = []
        for i in range(randomRuns):
            data.append(np.array(DataGenerator.getCascadeDataBrainpy(matrix, samples, 0, verbose)).T)
        
        for i in range(randomRuns):
            time_gcss = time.process_time()
            matrixGCSS = GCSS.gcss(data[i].T, alpha, tauMax[0], returnAll=False)
            end_gcss = time.process_time()
            runtimes[i,0] = end_gcss - time_gcss
        
        for i in range(randomRuns):
            time_lkif = time.process_time()
            matrixLKIF = LKIF.lkif(data[i].T, alpha, tau_max=tauMax[1])
            end_lkif = time.process_time()
            runtimes[i,1] = end_lkif - time_lkif
        
        for i in range(randomRuns):
            time_pcmci = time.process_time()
            matrixPCMCI, p_values = PCMCI.PCMCIPlus(data[i], [], range(data[i].shape[1]), None, tauMax[2], alpha if alpha<=1 else 1, contempLinks=True)
            end_pcmci = time.process_time()
            runtimes[i,2] = end_pcmci - time_pcmci
        
        np.save(data_dir + "/runtimes.npy", runtimes)
    else:
        runtimes = np.load(data_dir + "/runtimes.npy")
    from matplotlib import pyplot as plt
    plt.figure(figsize=(5,4), layout="constrained")
    plt.boxplot(runtimes[:,::-1], labels=["PCMCI","LKIF", "GCSS"], patch_artist=True)
    plt.ylabel("Runtime (s)", fontsize=14)
    plt.yscale("log")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(diag_dir + "/runtimes_withOutliers.png", dpi=300)

def noiseEvaluations(plotOnly, noiseScales, samples, alpha, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(noiseScales), randomRuns, 4, 3))
        seed = comm.Get_rank()
        truthMatrixVAR = mediumCouplingMatrixVAR_LowDense
        fullOutVAR = np.zeros((len(noiseScales), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(noiseScales)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        localFullVAR = []
        print("Node " + str(comm.Get_rank()) + " executing " + str(len(param_combs)) + " combinations for noise scale experiment")
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = samples, alpha= alpha, couplingStrength= couplingStrength, noiseScale= noiseScales[j], tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, metrics))

            metrics = getMetricOfRealization(couplingMatrix = truthMatrixVAR, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "VAR", 
                                                 samples = samples, alpha= alpha, couplingStrength= couplingStrength*0.1, noiseScale= noiseScales[j], tauMax= tauMax,
                                                  seed= seed, evalType= "Full", verbose=verbose)
            seed += comm.Get_size()
            localFullVAR.append((j, i, metrics))
        print("Node " + str(comm.Get_rank()) + " finished noise scale experiment")
        gathered = comm.gather(localFull, root=0)
        gatheredVAR = comm.gather(localFullVAR, root = 0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
            for result in gatheredVAR:
                for j, i, value in result:
                    fullOutVAR[j, i] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_Noise_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
            np.save(data_dir + "/VAR_6d_Noise_"+str(randomRuns)+"_runs_metrics.npy", fullOutVAR)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_Noise_"+str(randomRuns)+"_runs_metrics.npy")
        fullOutVAR = np.load(data_dir + "/VAR_6d_Noise_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, noiseScales, "", diag_dir + "/noise6dCascades", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Noise Scale σ", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        vis.saveMCCCurve(mean.T, noiseScales, "", diag_dir + "/noise6dCascades_Paper", stdDev.T, greyAxisAt=0.01, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Noise Scale σ", ylabel ="Matthews Correlation Coefficient", xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, figsize=(5.5,4))
        vis.saveMCCCurve(median.T, noiseScales, "", diag_dir + "/noise6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Noise Scale σ", ylabel ="Matthews Correlation Coefficient",xscale ="log")
        
        scores = MCCFromFull(fullOutVAR, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, noiseScales, "", diag_dir + "/noise6dVAR", stdDev.T, show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Noise Scale σ", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        vis.saveMCCCurve(median.T, noiseScales, "", diag_dir + "/noise6dVARQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, show=True, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Noise Scale σ", ylabel ="Matthews Correlation Coefficient", xscale ="log")
        
def delta_t_Evaluations(plotOnly, timeSteps, samples, alpha, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    if not plotOnly:
        truthMatrix = mediumCouplingMatrixCascade_LowDense
        fullOut = np.zeros((len(timeSteps), randomRuns, 4, 3))
        param_combs = list(product(np.arange(len(timeSteps)), np.arange(randomRuns)))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        print("Node " + str(comm.Get_rank()) + " executing " + str(len(param_combs)) + " combinations for time step experiment")
        for j,i in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = truthMatrix, algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = samples, alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose, deltaT=timeSteps[j])
            localFull.append((j, i, metrics))
        print("Node " + str(comm.Get_rank()) + " finished time step experiment")
        gathered = comm.gather(localFull, root=0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, value in result:
                    fullOut[j, i] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_Timestep_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_Timestep_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        # get central 80% of data
        median, lowerQ, higherQ = getMedianQuantile(scores, quantile=0.2, axis=1)
        vis.saveMCCCurve(mean.T, timeSteps, "", diag_dir + "/Timestep6dCascades", stdDev.T, 
                         show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Timestep Δt", ylabel ="Matthews Correlation Coefficient", 
                         xscale ="log")
        vis.saveMCCCurve(mean.T, timeSteps, "", diag_dir + "/Timestep6dCascades_Paper", stdDev.T, greyAxisAt=0.1, 
                         show=False, save = True, xlabel = "Timestep Δt", ylabel ="", yTickLabels=False, 
                         xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, figsize=(4.5,4))
        vis.saveMCCCurve(median.T, timeSteps, "", diag_dir + "/Timestep6dCascadesQuantiles", [], quantileLower = lowerQ.T, quantileHigher=higherQ.T, 
                         show=False, save = True, rowLabels=["GCSS", "LKIF", "PCMCI"], xlabel = "Timestep Δt", ylabel ="Matthews Correlation Coefficient",
                         xscale ="log")

def random_graph_Evaluations(plotOnly, couplingGraphs, sampleCounts, alpha, randomRuns, tauMax, couplingStrength, verbose = True, comm = None, data_dir = "./data", diag_dir = "./diagrams"):
    if not comm:
        comm = MPI.COMM_WORLD
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    defaultCols_with_alpha = [(*mcolors.to_rgba(c)[:3], 0.4) for c in mpl.color_sequences["tab10"]]
    if not plotOnly:
        fullOut = np.zeros((len(sampleCounts), randomRuns, 4, 3, len(couplingGraphs)))
        param_combs = list(product(np.arange(len(sampleCounts)), np.arange(randomRuns), np.arange(len(couplingGraphs))))
        param_combs = param_combs[comm.Get_rank()::comm.Get_size()]
        localFull = []
        print("Node " + str(comm.Get_rank()) + " executing " + str(len(param_combs)) + " combinations for random graph experiment")
        for j,i,k in param_combs:
            metrics = getMetricOfRealization(couplingMatrix = couplingGraphs[k], algorithms = ["GCSS", "LKIF", "PCMCI"], model= "Cascade", 
                                                samples = sampleCounts[j], alpha= alpha, couplingStrength= couplingStrength, noiseScale= 0.01, tauMax= tauMax,
                                                seed= 0, evalType= "Full", verbose=verbose)
            localFull.append((j, i, k, metrics))
        print("Node " + str(comm.Get_rank()) + " finished random graph experiment")
        gathered = comm.gather(localFull, root=0)
        if comm.Get_rank() == 0:
            for result in gathered:
                for j, i, k, value in result:
                    fullOut[j, i, :, :, k] = value
        if comm.Get_rank() == 0:
            np.save(data_dir + "/6d_Graphs_"+str(randomRuns)+"_runs_metrics.npy", fullOut)
    elif comm.Get_rank() == 0:
        fullOut = np.load(data_dir + "/6d_Graphs_"+str(randomRuns)+"_runs_metrics.npy")
    if comm.Get_rank() == 0:
        scores = MCCFromFull(fullOut, axis=2)
        mean, stdDev = getMeanStdDev(scores, axis = 1)
        fig = plt.figure(figsize=(4.5,4), layout ='constrained')
        for k in range(mean.shape[2]):
            if k == 0:
                vis.saveMCCCurve(mean[:,:,k].T, sampleCounts, "", diag_dir + "/6dCascades_Paper_Graph"+str(k), stdDev[:,:,k].T, 
                                 show=False, save = False, xlabel = "Number of Samples", ylabel ="", 
                                 yTickLabels=False, xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, fig = fig, 
                                 closePlot=False, alpha_fillBetween = 0.15)
            elif k != mean.shape[2]-1:
                vis.saveMCCCurve(mean[:,:,k].T, sampleCounts, "", diag_dir + "/6dCascades_Paper_Graph"+str(k), 
                                 show=False, save = False, colors = defaultCols_with_alpha[:3], xlabel = "Number of Samples", ylabel ="", 
                                 yTickLabels=False, xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, fig = fig, 
                                 closePlot=False, alpha_fillBetween = 0.15)
            else:
                vis.saveMCCCurve(mean[:,:,k].T, sampleCounts, "", diag_dir + "/6dCascades_Paper_RandomGraphs", greyAxisAt=1000, 
                                 show=False, save = True, colors = defaultCols_with_alpha[:3], xlabel = "Number of Samples", ylabel ="", 
                                 yTickLabels=False, xscale ="log", yLims=[-0.23, 1.03], fontsizeFactor=1.2, moveYLabel=-15, fig = fig, 
                                 closePlot=True, alpha_fillBetween = 0.15)


# MAIN

def main():
    comm = MPI.COMM_WORLD

    # my main experiment for the paper is in "./data/tauMax_1", as other tauMax settings disadvantaged the delay-sensitive LKIF method drastically.
    data_dir = "./data/tauMax_1"
    diag_dir = "./diagrams/tauMax_1"

    if comm.Get_rank() == 0:
        for directory in [data_dir, diag_dir]:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
            except PermissionError:
                print(f"Permission denied: Unable to create directory '{directory}'.")
            except Exception as e:
                print(f"An error occurred trying to create '{directory}': {e}")
    
    # set this parameter to True if you only want to plot results from the .npy data files.
    plotOnly = False
    randomRuns = 100
    alpha = 0.05
    samples = 1000
    couplingStrength = 1
    tauMax = [1,1,1]

    # run the random graph generation on a separate earlier run, it's not parallelizable.
    # generate_random_graphs()
    graphs = get_random_graphs()

    random_graph_Evaluations(plotOnly = plotOnly,
    couplingGraphs=graphs,
    sampleCounts = [50, 100, 200, 500, 1000, 2000, 5000, 10000],
    alpha = alpha,
    couplingStrength= couplingStrength,
    tauMax = tauMax,
    randomRuns = randomRuns, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Random-Graph Evaluation finished")

    exit()

    system6dEvaluations(plotOnly = plotOnly,
    alpha = alpha,
    samples = samples,
    randomRuns = randomRuns,
    tauMax = tauMax,
    couplingStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("System Size/Density Evaluation finished")

    sampleEvaluations(plotOnly = plotOnly,
    sampleCounts = [50, 100, 200, 500, 1000, 2000, 5000, 10000],
    alpha = alpha,
    randomRuns = randomRuns,
    tauMax = tauMax,
    couplingStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Sample Evaluation finished")
    
    couplStrength6dEvaluations(plotOnly = plotOnly,
    couplStrengths = np.array([0.01, 0.02, 0.05, 0.07,0.1,0.15,0.2,0.25,0.3]),
    samples = samples,
    alpha = alpha,
    randomRuns = randomRuns,
    tauMax = tauMax, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Coupling Strength Evaluation finished")

    delta_t_Evaluations(plotOnly = plotOnly,
    timeSteps=[0.01, 0.03, 0.1, 0.3, 1],
    samples = samples,
    alpha = alpha,
    tauMax = tauMax,
    randomRuns = randomRuns,
    couplingStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Time Step Evaluation finished")

    nonStationaryExp(plotOnly = plotOnly,
    ceilings = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    alpha = alpha,
    samples = samples,
    tauMax = tauMax,
    randomRuns = randomRuns, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Non-Stationarity Evaluation finished")

    noiseEvaluations(plotOnly = plotOnly,
    noiseScales= [0.005, 0.01, 0.02, 0.05, 0.1, 0.2,0.5],
    samples = samples,
    alpha = alpha,
    tauMax = tauMax,
    randomRuns = randomRuns,
    couplingStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Noise Evaluation finished")

    delay6dEvaluations(plotOnly = plotOnly,
    delaySizes = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0, 1.5,2.0,3.0],
    samples = samples,
    alpha = alpha,
    randomRuns = randomRuns,
    couplStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Delay Evaluation finished")

    runtimeEvaluations(plotOnly = plotOnly,
    samples=samples,
    matrix=mediumCouplingMatrixCascade_LowDense,
    alpha = alpha,
    randomRuns = randomRuns,
    tauMax = tauMax,
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Runtime Evaluation finished")
    
    sampleEvaluationsAppendix(plotOnly = plotOnly,
    sampleCounts = [50, 100, 200, 500, 1000, 2000],
    alpha = alpha,
    randomRuns = randomRuns,
    tauMax = tauMax,
    couplingStrength=couplingStrength, 
    verbose=False,
    comm=comm,
    data_dir=data_dir,
    diag_dir=diag_dir)
    if comm.Get_rank() == 0: print("Sample Evaluation finished")

    GCSS.close_octave()

if __name__ == "__main__":
    main()