import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from progress.bar import Bar
import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

def dynSysX3(t,y,A,B,C,D,deltaT,noiseArr):
    return A*(np.power(y,3)) + B*y + C + D@(y-1) + noiseArr[int(t/deltaT)%len(noiseArr)]

def gen3dDelayData(couplingMatrix, delayMatrix, timeSteps = 1000, savename = "threeInteractions", verbose = True):
    
    
    if couplingMatrix.shape != (3,3) or delayMatrix.shape != (3,3):
        print("Error, coupling or delay matrix not of shape (3,3)")
        return
    dt = 0.001
    maxDelay = np.max(delayMatrix)
    xdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    ydelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    zdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')

    f = lambda x,y,z,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (ydelay(t-delayMatrix[0,1])-1) + couplingMatrix[0,2] * (zdelay(t-delayMatrix[0,2])-1),\
                          - pow(y, 3) + y + couplingMatrix[1,0] * (xdelay(t-delayMatrix[1,0])-1) + couplingMatrix[1,2] * (zdelay(t-delayMatrix[1,2])-1), \
                            - pow(z, 3) + z + couplingMatrix[2,0] * (xdelay(t-delayMatrix[2,0])-1) + couplingMatrix[2,1] * (ydelay(t-delayMatrix[2,1])-1))
    
    g = lambda x,y,z,t: (0.01, 0.01, 0.01)

    integral = bp.sdeint(f, g, state_delays={'x':xdelay, 'y': ydelay, 'z':zdelay})

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z'],
                                inits = [1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)

    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100])).reshape((3,-1))
    return stacked

def gen3dNoDelayData(couplingMatrix, timeSteps = 1000, verbose = True):
    
    dt = 0.001
    f = lambda x,y,z,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (y-1) + couplingMatrix[0,2] * (z-1),\
                          - pow(y, 3) + y + couplingMatrix[1,0] * (x-1) + couplingMatrix[1,2] * (z-1), \
                           - pow(z, 3) + z + couplingMatrix[2,0] * (x-1) + couplingMatrix[2,1] * (y-1))
    g = lambda x,y,z,t: (0.01, 0.01, 0.01)

    integral = bp.sdeint(f, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z'],
                                inits = [1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100])).reshape((3,-1))
    return stacked

# 6 dimensional cubic diff. eq. with delays
def gen6dDelayData(couplingMatrix, delayMatrix, timeSteps = 1000, verbose = True):
    
    if couplingMatrix.shape != (6,6) or delayMatrix.shape != (6,6):
        print("Error, coupling or delay matrix not of shape (6,6)")
        return
    dt = 0.001
    maxDelay = np.max(delayMatrix)
    xdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    ydelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    zdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    adelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    bdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')
    cdelay = bm.TimeDelay(bm.ones(1), maxDelay, before_t0=1., dt=dt, interp_method='linear_interp')

    f = lambda x,y,z,a,b,c,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (ydelay(t-delayMatrix[0,1])-1) + couplingMatrix[0,2] * (zdelay(t-delayMatrix[0,2])-1) + couplingMatrix[0,3] * (adelay(t-delayMatrix[0,3])-1)+ couplingMatrix[0,4] * (bdelay(t-delayMatrix[0,4])-1)+ couplingMatrix[0,5] * (cdelay(t-delayMatrix[0,5])-1),\
                            - pow(y, 3) + y + couplingMatrix[1,0] * (xdelay(t-delayMatrix[1,0])-1)+ couplingMatrix[1,2] * (zdelay(t-delayMatrix[1,2])-1) + couplingMatrix[1,3] * (adelay(t-delayMatrix[1,3])-1)+ couplingMatrix[1,4] * (bdelay(t-delayMatrix[1,4])-1)+ couplingMatrix[1,5] * (cdelay(t-delayMatrix[1,5])-1),\
                            - pow(z, 3) + z + couplingMatrix[2,0] * (xdelay(t-delayMatrix[2,0])-1)+ couplingMatrix[2,1] * (ydelay(t-delayMatrix[2,1])-1) + couplingMatrix[2,3] * (adelay(t-delayMatrix[2,3])-1)+ couplingMatrix[2,4] * (bdelay(t-delayMatrix[2,4])-1)+ couplingMatrix[2,5] * (cdelay(t-delayMatrix[2,5])-1),\
                            - pow(a, 3) + a + couplingMatrix[3,0] * (xdelay(t-delayMatrix[3,0])-1)+ couplingMatrix[3,1] * (ydelay(t-delayMatrix[3,1])-1) + couplingMatrix[3,2] * (zdelay(t-delayMatrix[3,2])-1) + couplingMatrix[3,4] * (bdelay(t-delayMatrix[3,4])-1)+ couplingMatrix[3,5] * (cdelay(t-delayMatrix[3,5])-1),\
                            - pow(b, 3) + b + couplingMatrix[4,0] * (xdelay(t-delayMatrix[4,0])-1)+ couplingMatrix[4,1] * (ydelay(t-delayMatrix[4,1])-1) + couplingMatrix[4,2] * (zdelay(t-delayMatrix[4,2])-1) + couplingMatrix[4,3] * (adelay(t-delayMatrix[4,3])-1)+  couplingMatrix[4,5] * (cdelay(t-delayMatrix[4,5])-1),\
                            - pow(c, 3) + c + couplingMatrix[5,0] * (xdelay(t-delayMatrix[5,0])-1)+ couplingMatrix[5,1] * (ydelay(t-delayMatrix[5,1])-1) + couplingMatrix[5,2] * (zdelay(t-delayMatrix[5,2])-1) + couplingMatrix[5,3] * (adelay(t-delayMatrix[5,3])-1)+ couplingMatrix[5,4] * (bdelay(t-delayMatrix[5,4])-1)\
                            )
    
    g = lambda x,y,z,a,b,c,t: (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

    integral = bp.sdeint(f, g, state_delays={'x':xdelay, 'y': ydelay, 'z':zdelay, 'a':adelay, 'b':bdelay, 'c':cdelay})

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c'],
                                inits = [1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100],runner.mon.a[::100], runner.mon.b[::100],runner.mon.c[::100])).reshape((6,-1))
    return stacked

def gen6dNoDelayData(couplingMatrix, timeSteps = 1000, verbose = True):

    if couplingMatrix.shape != (6,6):
        print("Error, coupling or delay matrix not of shape (6,6)")
        return
    dt = 0.001

    f = lambda x,y,z,a,b,c,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (y-1) + couplingMatrix[0,2] * (z-1) + couplingMatrix[0,3] * (a-1)+ couplingMatrix[0,4] * (b-1)+ couplingMatrix[0,5] * (c-1),\
                            - pow(y, 3) + y + couplingMatrix[1,0] * (x -1)+ couplingMatrix[1,2] * (z -1) + couplingMatrix[1,3] * (a -1)+ couplingMatrix[1,4] * (b -1)+ couplingMatrix[1,5] * (c -1),\
                            - pow(z, 3) + z + couplingMatrix[2,0] * (x -1)+ couplingMatrix[2,1] * (y -1) + couplingMatrix[2,3] * (a -1)+ couplingMatrix[2,4] * (b -1)+ couplingMatrix[2,5] * (c -1),\
                            - pow(a, 3) + a + couplingMatrix[3,0] * (x -1)+ couplingMatrix[3,1] * (y -1) + couplingMatrix[3,2] * (z -1) + couplingMatrix[3,4] * (b -1)+ couplingMatrix[3,5] * (c -1),\
                            - pow(b, 3) + b + couplingMatrix[4,0] * (x -1)+ couplingMatrix[4,1] * (y -1) + couplingMatrix[4,2] * (z -1) + couplingMatrix[4,3] * (a -1)+  couplingMatrix[4,5] * (c -1),\
                            - pow(c, 3) + c + couplingMatrix[5,0] * (x -1)+ couplingMatrix[5,1] * (y -1) + couplingMatrix[5,2] * (z -1) + couplingMatrix[5,3] * (a -1)+ couplingMatrix[5,4] * (b -1)\
                            )
    
    g = lambda x,y,z,a,b,c,t: (0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

    integral = bp.sdeint(f, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c'],
                                inits = [1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100],runner.mon.a[::100], runner.mon.b[::100],runner.mon.c[::100])).reshape((6,-1))
    return stacked

def gen12dNoDelayData(couplingMatrix, timeSteps = 1000, verbose = True):
    
    if couplingMatrix.shape != (12,12):
        print("Error, coupling or delay matrix not of shape (12,12)")
        return
    dt = 0.001

    def function12d(x,y,z,a,b,c,d,e,f,g,h,j,t):
        values = jnp.stack([x,y,z,a,b,c,d,e,f,g,h,j])
        minus1 = values-1
        result = -jnp.power(values, 3) + values + couplingMatrix @ minus1 
        return result[0], result[1], result[2], result[3], result[4], result[5],result[6], result[7], result[8], result[9], result[10], result[11]
    g = lambda x,y,z,a,b,c,d,e,f,g,h,j,t: (0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

    integral = bp.sdeint(function12d, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c', 'd','e','f','g','h','j'],
                                inits = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100],runner.mon.a[::100], runner.mon.b[::100],runner.mon.c[::100],runner.mon.d[::100], runner.mon.e[::100],runner.mon.f[::100],runner.mon.g[::100], runner.mon.h[::100],runner.mon.j[::100])).reshape((12,-1))
    return stacked

def gen3dAutoCorrData(couplingMatrix, autoCorr, verbose = True):
    
    dt = 0.001
    f = lambda x,y,z,t: (- autoCorr*pow(x, 3) + autoCorr*x + couplingMatrix[0,1] * (y-1) + couplingMatrix[0,2] * (z-1),\
                          - autoCorr*pow(y, 3) + autoCorr*y + couplingMatrix[1,0] * (x-1) + couplingMatrix[1,2] * (z-1), \
                           - autoCorr*pow(z, 3) + autoCorr*z + couplingMatrix[2,0] * (x-1) + couplingMatrix[2,1] * (y-1))
    g = lambda x,y,z,t: (0.01, 0.01, 0.01)

    integral = bp.sdeint(f, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z'],
                                inits = [1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(100.)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100])).reshape((3,-1))
    return stacked

def getCascadeDataBrainpy(couplingMatrix, samples, delay = 0, verbose = True):
    if couplingMatrix.shape != ((3,3)) and couplingMatrix.shape != ((6,6)) and couplingMatrix.shape != ((12,12)):
        print("Error, coupling matrix not of shape (3,3) or (6,6) or (12,12)")
        return
    if delay == 0:
        if couplingMatrix.shape == (3,3):
            return gen3dNoDelayData(couplingMatrix, samples, verbose)
        elif couplingMatrix.shape == (6,6):
            return gen6dNoDelayData(couplingMatrix, samples, verbose)
        elif couplingMatrix.shape == (12,12):
            return gen12dNoDelayData(couplingMatrix,samples, verbose)
    else:
        if couplingMatrix.shape == (3,3):
            return gen3dDelayData(couplingMatrix, np.ones((3,3)) * delay, samples, verbose)
        elif couplingMatrix.shape == (6,6):
            return gen6dDelayData(couplingMatrix, np.ones((6,6)) * delay, samples, verbose)
    print("Error, no implementation of delayed 12x12 matrix")
    return

def getCascade3dConfoundedBrainpy(confounderFct, couplingMatrix, samples, forcingNoise = 0.01, verbose = True):
    
    if couplingMatrix.shape != (3,3):
        print("Error, coupling matrix not of shape (3,3)")
        return
    dt = 0.001
    def function3dConfounded(x,y,z,c,t):
        values = jnp.stack([x,y,z])
        minus1 = values-1
        result = -jnp.power(values, 3) + values + couplingMatrix @ minus1 + c
        return result[0], result[1], result[2], confounderFct(c,t)
    g = lambda x,y,z,c,t: (0.01, 0.01, 0.01, forcingNoise)

    integral = bp.sdeint(function3dConfounded, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z', 'c'],
                                inits = [1.,1.,1.,0.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(samples / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100], runner.mon.c[::100])).reshape((4,-1))
    return stacked

def getCascade2dConfoundedBrainpy(confounderFct, couplingMatrix, samples, forcingNoise = 0.01, verbose = True):
    
    if couplingMatrix.shape != (2,2):
        print("Error, coupling matrix not of shape (2,2)")
        return
    dt = 0.001
    def function2dConfounded(x,y,c,t):
        values = jnp.stack([x,y])
        minus1 = values-1
        result = -jnp.power(values, 3) + values + couplingMatrix @ minus1 + c
        return result[0], result[1], confounderFct(c,t)
    g = lambda x,y,c,t: (0.01, 0.01, forcingNoise)

    integral = bp.sdeint(function2dConfounded, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'c'],
                                inits = [1.,1.,0.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(samples / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100], runner.mon.c[::100])).reshape((3,-1))
    return stacked

def getCascade6dConfoundedBrainpy(confounderFct, couplingMatrix, samples, forcingNoise = 0.01, verbose = True):
    
    if couplingMatrix.shape != (6,6):
        print("Error, coupling matrix not of shape (6,6)")
        return
    dt = 0.001
    def function6dConfounded(x,y,z,a,b,c,d,t):
        values = jnp.stack([x,y,z,a,b,c])
        minus1 = values-1
        result = -jnp.power(values, 3) + values + couplingMatrix @ minus1 + d
        return result[0], result[1], result[2], result[3], result[4], result[5], confounderFct(d,t)
    g = lambda x,y,z,a,b,c,d,t: (0.01, 0.01, 0.01,0.01, 0.01, 0.01, forcingNoise)

    integral = bp.sdeint(function6dConfounded, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z', 'a', 'b','c','d'],
                                inits = [1.,1.,1.,1.,1.,1.,0.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(samples / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100], 
                        runner.mon.a[::100], runner.mon.b[::100],runner.mon.c[::100], runner.mon.d[::100])).reshape((7,-1))
    return stacked

def getCascadeData(couplingMatrix, samples, deltaTOutput, noiseScale, seed, constantNoiseDuration = 1, deltaTSim = 0.001):
    n, _ = couplingMatrix.shape
    endT = samples*deltaTOutput
    timeSpan = [0, endT]
    timePoints = [x * deltaTSim for x in range(int(endT * (1/deltaTSim)))]
    randomGen = np.random.default_rng(seed)
    noiseArray = randomGen.normal(loc=0,scale=noiseScale,size=(int((1/constantNoiseDuration)*endT / deltaTOutput), n))
    init = np.ones(n)
    A = np.ones(n) * (-1)
    B = np.ones(n)
    C = np.zeros(n)
    sol = solve_ivp(dynSysX3, timeSpan, init, t_eval = timePoints, args=(A,B,C,couplingMatrix,deltaTOutput*constantNoiseDuration,noiseArray))
    return sol.y.T[::int(deltaTOutput/deltaTSim)]

def getVARData(couplingMatrix, samples, noiseScale, seed, delay=0):
    from tigramite.toymodels import structural_causal_processes as toys
    n,_ = couplingMatrix.shape
    randomGen = np.random.default_rng(seed)
    noiseArray = randomGen.normal(loc = 0, scale = noiseScale, size=(int(1.1*samples), n))
    links = {}
    def lin(x): return x
    for i in range(n):
        row = []
        for j in range(n):
            if couplingMatrix[i,j] != 0:
                if i == j:
                    row.append(((j, -1), couplingMatrix[i,j], lin))
                else:
                    row.append(((j, -1 - delay), couplingMatrix[i,j], lin))
        links[i] = row
    data_full, nonstationarity_indicator = toys.structural_causal_process(
        links=links, T=samples, noises=noiseArray, seed=seed, transient_fraction=0.1)
    assert not nonstationarity_indicator
    return data_full

# corner and border treatment are a bit confusing - for "energy conservation" (or stability around startVal), one needs to use the following formula when calling:
# couplingGrid = couplingFactor, cornerAuto=1 - (couplingFactor * 2), borderAuto=1 - (couplingFactor * 3), borderLoss = -couplingFactor, autoCoupling = 1 - (couplingFactor*4), startVal=XY
def getGridData(gridX, gridY, couplingGrid, autoCoupling, driverCoupling, selfNoise, timeSteps, continuousGrid = False, 
                startVal = 0, cornerAuto = 0.5, borderAuto = 0.25, borderLoss = -0.25, gridToDriverCoupling=0, gridToDriverDelay=4, 
                driverAutoReg = 0, driverNoise=1, driverTimeSeries = [], driverNoiseSeries = [], seed = 0):
    randomGen = np.random.default_rng(seed)
    valuesGrid = startVal * np.ones((gridX * gridY, timeSteps))
    # to allow for energy dissipation to the border of the grid, basically modelling the grid to be surrounded by cells with constant value startVal
    constLoss = np.zeros((gridX * gridY))
    lossValue = - borderLoss * startVal
    generateDriver = False
    generateNoise = False
    if driverTimeSeries == []:
        driverTimeSeries = np.zeros(timeSteps)
        if driverNoiseSeries == []:
            driverNoiseSeries = np.zeros(timeSteps)
            generateNoise = True
        generateDriver = True
        
    couplingMatrix = np.zeros((gridX*gridY, gridX * gridY))
    # could speed this up by writing more elegantly, whatever
    if continuousGrid:
        for i in range(gridX*gridY):
            couplingMatrix[i, i-gridX] = couplingGrid
            couplingMatrix[i,(i+gridX)%(gridX*gridY)] = couplingGrid
            couplingMatrix[i,i-1] = couplingGrid
            couplingMatrix[i,(i+1)%(gridX*gridY)] = couplingGrid
    else:
        for i in range(gridX*gridY):
            if i == 0 or i == gridX-1 or i == gridX * gridY - 1 or i == gridX * (gridY-1):
                couplingMatrix[i,i] = cornerAuto + (2 * borderLoss)
                constLoss[i] = lossValue * 2
            elif i < gridX or i % gridX == 0 or i%gridX == gridX-1 or i > gridX * (gridY-1):
                couplingMatrix[i,i] = borderAuto + borderLoss
                constLoss[i] = lossValue
            else:
                couplingMatrix[i,i] = autoCoupling
            if i >= gridX:
                couplingMatrix[i, i-gridX] = couplingGrid
            if i < (gridX*gridY)-gridX:
                couplingMatrix[i,i+gridX] = couplingGrid
            if i % gridX > 0:
                couplingMatrix[i,i-1] = couplingGrid
            if i % gridX != (gridX-1):
                couplingMatrix[i,i+1] = couplingGrid
    driverCoupling = np.ones(gridX*gridY) * driverCoupling
    for j in range(1,timeSteps):
        valuesGrid[:,j] = couplingMatrix@valuesGrid[:,j-1] + constLoss + driverCoupling * (driverTimeSeries[j-1] - startVal) + randomGen.normal(loc = 0, scale = selfNoise, size=gridX*gridY)
        if generateDriver:
            driverTimeSeries[j] = driverAutoReg * driverTimeSeries[j-1] + driverNoiseSeries[j] + (randomGen.normal(loc=0,scale=driverNoise) if generateNoise else 0) + (0 if j <= (gridToDriverDelay-1) else np.sum(gridToDriverCoupling * valuesGrid[:,j-gridToDriverDelay]))
    if np.max(driverTimeSeries) > 100 or np.max(valuesGrid) > 100 or np.any(np.isnan(driverTimeSeries)) or np.any(np.isnan(valuesGrid)):
        print("error: values larger than 100 indicate lack of convergence / parametrization fault")
        exit()
    return driverTimeSeries, valuesGrid

# returns a multi-dim array that contains 1 for instability and 0 for stability in cascade data gen
def getStabilityMatrix(couplingMatrices, sampleCounts, couplingStrengths, noiseScales, runs):
    result = np.zeros((len(couplingMatrices), len(sampleCounts), len(couplingStrengths), len(noiseScales)))
    totalRuns = (len(couplingMatrices) * len(sampleCounts) * len(couplingStrengths) * len(noiseScales)) * runs
    progressBar = Bar('Processing', max=totalRuns)
    for matInd in range(len(couplingMatrices)):
        for sampInd in range(len(sampleCounts)):
            for strength in range(len(couplingStrengths)):
                matrix = couplingStrengths[strength] * couplingMatrices[matInd]
                for i in range(matrix.shape[0]):
                    matrix[i,i] = couplingMatrices[matInd][i,i]
                for noiseInd in range(len(noiseScales)):
                    for run in range(runs):
                        progressBar.next()
                        data = getCascadeData(couplingMatrices[matInd], sampleCounts[sampInd], 0.1, noiseScales[noiseInd], run, deltaTSim=0.01)
                        if np.min(data) <= -1:
                            result[matInd, sampInd, strength, noiseInd] = 1
                            print("Unstable config for matrix " + str(matInd) + ", " + str(sampleCounts[sampInd]) + " samples, coupling strength " + str(couplingStrengths[strength]) + " and noise scale " + str(noiseScales[noiseInd]))
    progressBar.finish()
    return result

def checkStability():
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
    highDensityCouplingMatrixVAR = np.array([
                                        [0.5,0,-1,-1,0,0],
                                        [1,0.5,0,0,0,0],
                                        [1,1,0.5,0,0,0],
                                        [0,0,0,0.5,-1,-1],
                                        [0,1,-1,0,0.5,1],
                                        [0,0,0,1,-1,0.5]])
    highDensityCouplingMatrixCascade = np.array([
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
    defaultCouplingMatrixCascade = np.array([[0,1,0],[-1,0,0],[1,-1,0]])
    for i in [mediumCouplingMatrixVAR, largeCouplingMatrixVAR, highDensityCouplingMatrixVAR, defaultCouplingMatrixVAR_LowDense, defaultCouplingMatrixVAR_HighDense, largeCouplingMatrixVAR_LowDense,largeCouplingMatrixVAR_HighDense]:
        matrix = 0.1 * i
        for m in range(matrix.shape[0]):
            matrix[m,m] = i[m,m]
        #data = getVARData(matrix, 2000, 0.1, 0)
        #plt.figure()
        #for j in range(data.shape[1]):
        #    plt.plot(data[:,j])
    #for i in [mediumCouplingMatrixCascade, largeCouplingMatrixCascade, highDensityCouplingMatrixCascade,defaultCouplingMatrixCascade_LowDense,defaultCouplingMatrixCascade_HighDense,largeCouplingMatrixCascade_LowDense,largeCouplingMatrixCascade_HighDense ]:
    for i in [defaultCouplingMatrixCascade]: 
        data = getCascadeData(3*i, 10000, 0.1,0.1, 0)
        plt.figure()
        for j in range(data.shape[1]):
            plt.plot(data[:,j], label = str(j))
        plt.legend()
    plt.show()

if __name__ == "__main__":
    # truthMatrix = np.array([[0.5,0.2,0],[-0.2,0.5,0],[0.2,-0.2,0.5]])
    # data = getVARData(truthMatrix, 1000, 0.01, 0, 5)
    # plt.plot(data)
    # plt.figure()
    # data2 = getVARData(truthMatrix, 1000, 0.01, 0, 0)
    # plt.plot(data2)
    # plt.plot(data[:,1] - data2[:,1])
    # plt.show()
    truthMatrix = np.array([[0,0,0,0,0,0],
                            [1,0,0,0,0,0],
                            [0,-1,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,1,0,0,1],
                            [0,0,0,0,-1,0]])
    data = getCascadeDataBrainpy(truthMatrix, 100)
    print(data.shape)
    fig, axs = plt.subplots(6,1, figsize=(3,6), sharex=True)
    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_yticks([])

    for i in range(data.shape[0]):
        axs[i].plot(data[i], color="black")
        #if i < 5:
            #ax.spines['bottom'].set_linewidth(2)
            #ax.spines['bottom'].set_color('black')
    plt.subplots_adjust(hspace=0)
    plt.savefig("diagrams/plainData.png", dpi=300)