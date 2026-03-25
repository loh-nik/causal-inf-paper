import numpy as np
from matplotlib import pyplot as plt

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

# Individual sized Delayed / Stochastic Differential Equation data generation
def gen3dDelayData(couplingMatrix, delayMatrix, timeSteps = 1000, noiseScale = 0.01, savename = "threeInteractions", verbose = True):
    
    
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
    
    g = lambda x,y,z,t: (noiseScale,) * 3

    integral = bp.sdeint(f, g, state_delays={'x':xdelay, 'y': ydelay, 'z':zdelay})

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z'],
                                inits = [1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)

    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100])).reshape((3,-1))
    return stacked

def gen3dNoDelayData(couplingMatrix, timeSteps = 1000, noiseScale = 0.01, verbose = True, deltaT = 0.1):
    
    dt = 0.001
    sampleSkip = int(deltaT/dt)
    f = lambda x,y,z,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (y-1) + couplingMatrix[0,2] * (z-1),\
                          - pow(y, 3) + y + couplingMatrix[1,0] * (x-1) + couplingMatrix[1,2] * (z-1), \
                           - pow(z, 3) + z + couplingMatrix[2,0] * (x-1) + couplingMatrix[2,1] * (y-1))
    g = lambda x,y,z,t: (noiseScale,) * 3

    integral = bp.sdeint(f, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z'],
                                inits = [1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / (1/deltaT))
    stacked = np.stack((runner.mon.x[::sampleSkip], runner.mon.y[::sampleSkip],runner.mon.z[::sampleSkip])).reshape((3,-1))
    return stacked

def gen6dDelayData(couplingMatrix, delayMatrix, timeSteps = 1000, noiseScale = 0.01, verbose = True):
    
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
    
    g = lambda x,y,z,a,b,c,t: (noiseScale,) * 6

    integral = bp.sdeint(f, g, state_delays={'x':xdelay, 'y': ydelay, 'z':zdelay, 'a':adelay, 'b':bdelay, 'c':cdelay})

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c'],
                                inits = [1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / 10)
    stacked = np.stack((runner.mon.x[::100], runner.mon.y[::100],runner.mon.z[::100],runner.mon.a[::100], runner.mon.b[::100],runner.mon.c[::100])).reshape((6,-1))
    return stacked

def gen6dNoDelayData(couplingMatrix, timeSteps = 1000, noiseScale = 0.01, verbose = True, deltaT = 0.1):

    if couplingMatrix.shape != (6,6):
        print("Error, coupling or delay matrix not of shape (6,6)")
        return
    dt = 0.001
    sampleSkip = int(deltaT/dt)
    f = lambda x,y,z,a,b,c,t: (- pow(x, 3) + x + couplingMatrix[0,1] * (y-1) + couplingMatrix[0,2] * (z-1) + couplingMatrix[0,3] * (a-1)+ couplingMatrix[0,4] * (b-1)+ couplingMatrix[0,5] * (c-1),\
                            - pow(y, 3) + y + couplingMatrix[1,0] * (x -1)+ couplingMatrix[1,2] * (z -1) + couplingMatrix[1,3] * (a -1)+ couplingMatrix[1,4] * (b -1)+ couplingMatrix[1,5] * (c -1),\
                            - pow(z, 3) + z + couplingMatrix[2,0] * (x -1)+ couplingMatrix[2,1] * (y -1) + couplingMatrix[2,3] * (a -1)+ couplingMatrix[2,4] * (b -1)+ couplingMatrix[2,5] * (c -1),\
                            - pow(a, 3) + a + couplingMatrix[3,0] * (x -1)+ couplingMatrix[3,1] * (y -1) + couplingMatrix[3,2] * (z -1) + couplingMatrix[3,4] * (b -1)+ couplingMatrix[3,5] * (c -1),\
                            - pow(b, 3) + b + couplingMatrix[4,0] * (x -1)+ couplingMatrix[4,1] * (y -1) + couplingMatrix[4,2] * (z -1) + couplingMatrix[4,3] * (a -1)+  couplingMatrix[4,5] * (c -1),\
                            - pow(c, 3) + c + couplingMatrix[5,0] * (x -1)+ couplingMatrix[5,1] * (y -1) + couplingMatrix[5,2] * (z -1) + couplingMatrix[5,3] * (a -1)+ couplingMatrix[5,4] * (b -1)\
                            )
    
    g = lambda x,y,z,a,b,c,t: (noiseScale,) * 6

    integral = bp.sdeint(f, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c'],
                                inits = [1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / (1/deltaT))
    stacked = np.stack((runner.mon.x[::sampleSkip], runner.mon.y[::sampleSkip],runner.mon.z[::sampleSkip],runner.mon.a[::sampleSkip], runner.mon.b[::sampleSkip],runner.mon.c[::sampleSkip])).reshape((6,-1))
    return stacked

def gen12dNoDelayData(couplingMatrix, timeSteps = 1000, noiseScale = 0.01, verbose = True, deltaT = 0.1):
    
    if couplingMatrix.shape != (12,12):
        print("Error, coupling or delay matrix not of shape (12,12)")
        return
    dt = 0.001
    sampleSkip = int(deltaT/dt)
    def function12d(x,y,z,a,b,c,d,e,f,g,h,j,t):
        values = jnp.stack([x,y,z,a,b,c,d,e,f,g,h,j])
        minus1 = values-1
        result = -jnp.power(values, 3) + values + couplingMatrix @ minus1 
        return result[0], result[1], result[2], result[3], result[4], result[5],result[6], result[7], result[8], result[9], result[10], result[11]
    g = lambda x,y,z,a,b,c,d,e,f,g,h,j,t: (noiseScale,) * 12

    integral = bp.sdeint(function12d, g)

    runner = bp.IntegratorRunner(integral,
                                monitors=['x', 'y', 'z','a', 'b', 'c', 'd','e','f','g','h','j'],
                                inits = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
                                dt=dt,
                                progress_bar=verbose)
    runner.run(timeSteps / (1/deltaT))
    stacked = np.stack((runner.mon.x[::sampleSkip], runner.mon.y[::sampleSkip],runner.mon.z[::sampleSkip],runner.mon.a[::sampleSkip], runner.mon.b[::sampleSkip],runner.mon.c[::sampleSkip],runner.mon.d[::sampleSkip], runner.mon.e[::sampleSkip],runner.mon.f[::sampleSkip],runner.mon.g[::sampleSkip], runner.mon.h[::sampleSkip],runner.mon.j[::sampleSkip])).reshape((12,-1))
    return stacked

# Chooses the correct implementation of a DDE/SDE
def getCascadeDataBrainpy(couplingMatrix, samples, noiseScale, delay = 0, verbose = True, deltaT = 0.1):
    if delay == 0:
        if couplingMatrix.shape == (3,3):
            return gen3dNoDelayData(couplingMatrix, samples, noiseScale, verbose, deltaT)
        elif couplingMatrix.shape == (6,6):
            return gen6dNoDelayData(couplingMatrix, samples, noiseScale, verbose, deltaT)
        elif couplingMatrix.shape == (12,12):
            return gen12dNoDelayData(couplingMatrix,samples, noiseScale, verbose, deltaT)
        elif couplingMatrix.shape[0] < 12 and couplingMatrix.shape[0] == couplingMatrix.shape[1]:
            # if the shapes don't match, just extend the matrix with isolated variables and throw them away afterwards. 
            # Since we found some limited accuracy degradation with vectorized operations, we would have to write a new method for every size.
            couplingMatrixNew = np.zeros((12,12))
            couplingMatrixNew[:couplingMatrix.shape[0], :couplingMatrix.shape[0]] = couplingMatrix
            return gen12dNoDelayData(couplingMatrixNew, samples, verbose, deltaT)[:couplingMatrix.shape[0]]
    else:
        if couplingMatrix.shape == (3,3):
            return gen3dDelayData(couplingMatrix, np.ones((3,3)) * delay, samples, noiseScale, verbose)
        elif couplingMatrix.shape == (6,6):
            return gen6dDelayData(couplingMatrix, np.ones((6,6)) * delay, samples, noiseScale, verbose)
    raise RuntimeError("No implementation of the given matrix shape")
    return

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

def checkRandomGraphStability():
    graphs = np.load("data/random_graphs.npy")
    for index, truthMatrix in enumerate(graphs):
        data = getCascadeDataBrainpy(truthMatrix, 10000, 0.05)
        fig, axs = plt.subplots(6,1, figsize=(3,6), sharex=True)
        for ax in axs:
            ax.set_xticklabels([])

        for i in range(data.shape[0]):
            axs[i].plot(data[i], color="black")
        plt.subplots_adjust(hspace=0)
        plt.savefig("diagrams/plainData"+str(index)+".png", dpi=300)

if __name__ == "__main__":
    checkRandomGraphStability()