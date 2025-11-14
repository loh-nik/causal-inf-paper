import numpy as np

def addMaskCSV():
    import pandas as pd
    df = pd.read_csv("DataSheet SeaIceAggregates.csv",sep=";")
    arr = np.ones(df.shape[0])
    for i in range(2,9):
        arr[i::12] = 0
    df["MarchSept"] = arr
    df.to_csv("DataSheet SeaIceAggregates2.csv", sep=";", index=False)

def amoc_to_ice(effect_strength):
    stdDevAMOC = 0.497
    stdDevIce = 135.58 / 3925.0
    convAMOC = 3.8
    print("Effect strength 1 Sv AMOC -> " + str((effect_strength*stdDevIce) / (convAMOC * stdDevAMOC)) + " sea ice conc.")

def ice_to_amoc(effect_strength):
    stdDevAMOC = 0.497
    stdDevIce = 135.58 / 3925.0
    convAMOC = 3.8
    print("Effect strength 0.01 sea ice conc -> " + str((effect_strength*(convAMOC*stdDevAMOC)) / (stdDevIce* 100)) + " Sv AMOC")

if __name__ == "__main__":
    # linear mediation calculations
    ice_to_amoc(-0.112)
    ice_to_amoc(0.026)
    amoc_to_ice(-0.054)
    