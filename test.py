import numpy as np
if __name__ == "__main__":
    mean = np.arange(36).reshape(6,2,3)
    print(mean)

    lowDense = mean[:3]
    highDense = mean[3:]
    mean = np.append(lowDense, highDense, axis=2)

    mean = np.append(mean[:,0,:], mean[:,1,:], axis=1)
    print(mean.shape)
    print(mean)
    print(mean[:,::3])

    print(np.array([0.01, 0.02, 0.05, 0.07,0.1,0.15,0.2,0.25,0.3])*10)