from matplotlib import pyplot as plt
import numpy as np

#import networkx as nx
import seaborn as sns

def saveBoxPlot(distributions):
    plt.boxplot(distributions)
    plt.show()

def saveROCCurve(TPR, FPR, values, title, filename, colors = [], rowLabels = [], show = False, save=True, annotateBest = True,
                 xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=300, xlabel ="", ylabel =""):
    fig = plt.figure(figsize=figsize, layout="constrained")
    #plt.suptitle(title, fontsize=15)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if TPR.shape != FPR.shape:
        print("Error: TPR and FPR shaped differently")
        exit()
    if TPR.shape[0] != len(values) and TPR.shape[1] != len(values):
        print("Value list doesn't match any dimension of data")
        exit()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05,1.05)
    # should be int for one-dimensional data, should be array of ints for two-dim data
    showIndices = np.argmin(np.power(1-TPR, 2) + np.power(FPR,2), axis=1 if len(TPR.shape) > 1 else 0)
    if len(TPR.shape) == 1:
        plt.plot(FPR, TPR)
        if annotateBest:
            plt.annotate(values[showIndices], (FPR[showIndices], TPR[showIndices]),fontsize=14)
    else:
        showLegend = True
        if len(rowLabels) == 0:
            rowLabels = np.zeros(TPR.shape[0])
            showLegend = False
        elif len(rowLabels) != TPR.shape[0]:
            print("Error: Not enough labels provided for TPR/FPR data")
            exit()
        if len(colors) > 0 and len(colors) != TPR.shape[0]:
            print("Error: Not enough colors provided for TPR/FPR data")
            exit()
        for i in range(TPR.shape[0]):
            if len(colors) == 0:
                plt.plot(FPR[i], TPR[i], label = str(rowLabels[i]))
                if annotateBest:
                    plt.plot(FPR[i, showIndices[i]], TPR[i, showIndices[i]], 'o', color="black")
            else:
                plt.plot(FPR[i], TPR[i], label = str(rowLabels[i]), color=colors[i])
                if annotateBest:
                    plt.plot(FPR[i, showIndices[i]], TPR[i, showIndices[i]], 'o', color=colors[i])
            if annotateBest:
                plt.annotate(values[showIndices[i]], (FPR[i, showIndices[i]], TPR[i, showIndices[i]]),fontsize=14)
        if showLegend:
            plt.legend(fontsize=14)
    if save:
        plt.savefig(filename,dpi = dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveMCCCurve(scores, values, title, filename, errors = [], colors = [], rowLabels=[], show = False, save= True, 
                xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=300, xlabel ="", ylabel ="", yAxisCut = False,
                  yAxisLinearLim = 1, quantileLower = [], quantileHigher = []):
    """Draws a plot with lines for data rows, optionally with error bars.
    \nScores are the MCC score values, in a 2D array with first dimension as the number of data lines drawn, \
    and second dimension equaling values dimension. 
    \nValues are drawn on the x-axis, e.g. number of samples"""
    fig = plt.figure(figsize=figsize, layout ='constrained')
    if scores.shape[0] != len(values) and scores.shape[1] != len(values):
        print("Value list doesn't match any dimension of data")
        exit()
    if len(errors) > 0 and errors.shape != scores.shape:
        print("Errors shape doesn't match scores shape")
        exit()
    #plt.suptitle(title, fontsize=15)
    plt.xscale(xscale)
    if yAxisCut:
        plt.yscale("symlog", linthresh = yAxisLinearLim)
        plt.yticks(np.append(np.arange(0,1,0.1), range(1,8)), labels=["0","","","","","0.5","","","","","1","","","","5","",""])
        ax = plt.gca()
        ax.grid(True, axis="y", color='lightgray', linestyle='-', linewidth=0.5)
        ax.axhline(y=yAxisLinearLim, color='gray', linewidth=1, linestyle='-')
    else:
        plt.yscale(yscale)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if len(scores.shape) == 1:
        plt.plot(values, scores)
        if len(errors) > 0:
            plt.fill_between(values, scores + errors, scores - errors, alpha = 0.2)
        elif len(quantileLower) > 0:
            plt.fill_between(values, quantileLower, quantileHigher, alpha = 0.2)
    else:
        showLegend = True
        if len(rowLabels) == 0:
            rowLabels = np.zeros(scores.shape[0])
            showLegend = False
        elif len(rowLabels) != scores.shape[0]:
            print("Error: Not enough labels provided for Score data")
            exit()
        if len(colors) > 0 and len(colors) != scores.shape[0]:
            print("Error: Not enough colors provided for Score data")
            exit()
        for i in range(scores.shape[0]):
            if len(colors) == 0:
                plt.plot(values, scores[i], label = str(rowLabels[i]))
                if len(errors) > 0:
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = 0.2)
                elif len(quantileLower) > 0:
                    plt.fill_between(values, quantileLower[i], quantileHigher[i], alpha = 0.2)
            else:
                plt.plot(values, scores[i], label = str(rowLabels[i]), color=colors[i])
                if len(errors) > 0:
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = 0.2, color= colors[i])
                elif len(quantileLower) > 0:
                    plt.fill_between(values, quantileLower[i], quantileHigher[i], alpha = 0.2, color= colors[i])
        if showLegend:
            plt.legend(fontsize=14)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveMCCCurveSubplot(subplotRows, subplotCols, scores, values, title, filename, errors = [], colors = [], rowLabels=[], show = False, save= True, 
                xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=300, xlabel ="", ylabel ="", yAxisCut = False,
                  yAxisLinearLim = 1, quantileLower = [], quantileHigher = [], share_x_axis = True, ylabelLoc = 1):
    """Draws subplots with lines for data rows, optionally with error bars.
    \nScores are the MCC score values, in a 3D array with first dimension as the number of subplots, second dim as data lines drawn, \
    and third dimension equaling values dimension. 
    \nValues are drawn on the x-axis, e.g. number of samples"""
    fig, axes = plt.subplots(subplotRows, subplotCols, figsize=figsize, layout ='constrained')
    if len(values) not in scores.shape:
        print("Value list doesn't match any dimension of data")
        exit()
    if len(errors) > 0 and errors.shape != scores.shape:
        print("Errors shape doesn't match scores shape")
        exit()
    #plt.suptitle(title, fontsize=15)
    for i, scoreTuple in enumerate(scores):
        ax = axes[i]
        if len(errors) > 0:
            errorTuple = errors[i]
        if len(colors) > 0:
            colorTuple = colors[i]
        if len(rowLabels) > 0:
            rowLabelTuple = rowLabels[i]
        ax.set_xscale(xscale)
        if yAxisCut:
            ax.set_yscale("symlog", linthresh = yAxisLinearLim)
            ax.set_yticks(np.append(np.arange(0,1,0.1), range(1,8)), labels=["0","","","","","0.5","","","","","1","","","","5","",""])
            ax.grid(True, axis="y", color='lightgray', linestyle='-', linewidth=0.5)
            ax.axhline(y=yAxisLinearLim, color='gray', linewidth=1, linestyle='-')
        else:
            ax.set_yscale(yscale)
        if share_x_axis and i < len(axes) - 1:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(xlabel, fontsize=14)
        if ylabelLoc < 0 or ylabelLoc == i:
            ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if len(scoreTuple.shape) == 1:
            ax.plot(values, scoreTuple)
            if len(errorTuple) > 0:
                ax.fill_between(values, scoreTuple + errorTuple, scoreTuple - errorTuple, alpha = 0.2)
            elif len(quantileLower) > 0:
                ax.fill_between(values, quantileLower, quantileHigher, alpha = 0.2)
        else:
            showLegend = True
            if len(rowLabels) == 0:
                rowLabelTuple = np.zeros(scoreTuple.shape[0])
                showLegend = False
            elif len(rowLabelTuple) != scoreTuple.shape[0]:
                print("Error: Not enough labels provided for Score data")
                exit()
            if len(colors) > 0 and len(colorTuple) != scoreTuple.shape[0]:
                print("Error: Not enough colors provided for Score data")
                exit()
            for i in range(scoreTuple.shape[0]):
                if len(colors) == 0:
                    ax.plot(values, scoreTuple[i], label = str(rowLabelTuple[i]))
                    if len(errors) > 0:
                        ax.fill_between(values, scoreTuple[i] + errorTuple[i], scoreTuple[i] - errorTuple[i], alpha = 0.2)
                    elif len(quantileLower) > 0:
                        ax.fill_between(values, quantileLower[i], quantileHigher[i], alpha = 0.2)
                else:
                    ax.plot(values, scoreTuple[i], label = str(rowLabelTuple[i]), color=colorTuple[i])
                    if len(errors) > 0:
                        ax.fill_between(values, scoreTuple[i] + errorTuple[i], scoreTuple[i] - errorTuple[i], alpha = 0.2, color= colorTuple[i])
                    elif len(quantileLower) > 0:
                        ax.fill_between(values, quantileLower[i], quantileHigher[i], alpha = 0.2, color= colorTuple[i])
            if showLegend:
                ax.legend(fontsize=14)
    # plt.subplots_adjust(hspace=0)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveMCCScatter(scores, values, title, filename, errors = [], colors = [], rowLabels=[], show = False, save= True, 
                xscale = "linear", yscale = "linear", figsize=(4.5,4), dpi=300, xlabel ="", ylabel ="", yAxisCut = False,
                  yAxisLinearLim = 1, quantileLower = [], quantileHigher = [], marker = '_', legend_outside = False, ylim = None):
    """Draws a plot with markers for data points, optionally with error bars.
    \nScores are the MCC score values, in a 2D array with first dimension as the number of data lines drawn, \
    and second dimension equaling values dimension. 
    \nValues are drawn on the x-axis, e.g. number of samples"""
    fig = plt.figure(figsize=figsize, layout ='constrained')
    if scores.shape[0] != len(values) and scores.shape[1] != len(values):
        print("Value list doesn't match any dimension of data")
        exit()
    if len(errors) > 0 and errors.shape != scores.shape:
        print("Errors shape doesn't match scores shape")
        exit()
    #plt.suptitle(title, fontsize=15)
    plt.xscale(xscale)
    if yAxisCut:
        plt.yscale("symlog", linthresh = yAxisLinearLim)
        plt.yticks(np.append(np.arange(0,1,0.1), range(1,8)), labels=["0","","","","","0.5","","","","","1","","","","5","",""])
        ax = plt.gca()
        ax.grid(True, axis="y", color='lightgray', linestyle='-', linewidth=0.5)
        ax.axhline(y=yAxisLinearLim, color='gray', linewidth=1, linestyle='-')
    else:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if len(scores.shape) == 1:
        if len(errors) > 0:
            plt.errorbar(values, scores, errors, fmt=marker, markersize=10, alpha = 0.3)
        elif len(quantileLower) > 0:
            plt.errorbar(values, scores, np.stack((quantileLower, quantileHigher), axis=0), fmt=marker, alpha = 0.3)
        else:
            plt.scatter(values, scores, marker=marker)
    else:
        showLegend = True
        if len(rowLabels) == 0:
            rowLabels = np.zeros(scores.shape[0])
            showLegend = False
        elif len(rowLabels) != scores.shape[0]:
            print("Error: Not enough labels provided for Score data")
            exit()
        if len(colors) > 0 and len(colors) != scores.shape[0]:
            print("Error: Not enough colors provided for Score data")
            exit()
        for i in range(scores.shape[0]):
            if len(colors) == 0:
                if len(errors) > 0:
                    plt.errorbar(values, scores[i], errors[i], fmt=marker, markersize=15, alpha = 0.3,label = str(rowLabels[i]))
                elif len(quantileLower) > 0:
                    plt.errorbar(values, scores[i], np.stack((quantileLower[i], quantileHigher[i]), axis=0), fmt=marker, alpha = 0.3,label = str(rowLabels[i]))
                else:
                    plt.scatter(values, scores[i],  marker=marker,label = str(rowLabels[i]))
            else:
                if len(errors) > 0:
                    plt.errorbar(values, scores[i],errors[i], fmt="none", markersize=15, alpha = 0.3, color= colors[i])
                    plt.errorbar(values, scores[i], fmt=marker, markersize=15, alpha = 1.0, color= colors[i],label = str(rowLabels[i]))
                    plt.plot(values, scores[i], color = colors[i], alpha=0.2)
                elif len(quantileLower) > 0:
                    plt.errorbar(values, scores[i], np.stack((quantileLower[i], quantileHigher[i]), axis=0), fmt=marker, alpha = 0.3, color= colors[i],label = str(rowLabels[i]))
                else:
                    plt.scatter(values, scores[i], label = str(rowLabels[i]), marker=marker, color=colors[i])
        if showLegend:
            if legend_outside:
                plt.legend(fontsize=14, bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                plt.legend(fontsize=14)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

def saveDecisionTree(data, labels, feature_names, class_names, filename, max_depth = 3, show = False, save = True):
    import dtreeviz
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    clf = clf.fit(data, labels)
    viz_model = dtreeviz.model(clf,
                                   X_train = data, y_train=labels,
                                   feature_names = feature_names,
                                   target_name = "Ranking", class_names = class_names)
    v = viz_model.view(fancy=False)
    if save:
        v.save(filename)
    if show:
        v.show()

def saveDecisionBoundaries(data, labels, filename, show = False, save = True):
    from dtreeviz import decision_boundaries
    from sklearn.tree import DecisionTreeClassifier
    if data.shape[1] != 2 :
        print("Invalid data to draw 2d decision boundaries")
        exit()
    if np.max(labels) == 0:
        print("Only one class found, no decision boundary can be drawn")
        return
    fix, ax = plt.subplots()
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf = clf.fit(data, labels)
    decision_boundaries(clf, data, labels, ax = ax)
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()

def saveCouplingMatrixGraph(matrix, title, filename, show = False, save= True, figsize=(4,4.5), dpi=300):
    from tigramite import plotting as tp
    ax = plt.subplots(1,1,layout="constrained", figsize=figsize)
    plt.title(title, fontsize=17)
    #G = nx.from_numpy_array(matrix, create_using =nx.DiGraph)
    matrixFull = np.zeros((matrix.shape[0], matrix.shape[1],2))
    matrixFull[:,:,1] = matrix
    tp.plot_graph(val_matrix=matrixFull,
            graph=matrixFull,
            show_colorbar=False,
            var_names=range(matrixFull.shape[0]),
            show_autodependency_lags=False,
            fig_ax = ax,
            node_aspect = 1,
            node_label_size=14,
            link_label_fontsize = 1
            )
    #G = nx.from_numpy_array(matrix, create_using =nx.DiGraph)
    #nx.draw(G, with_labels=True, font_weight='bold')
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()

def saveHeatmap(textValues, colorValues, title, filename, show=False, save=True, figsize = (4.5,4), dpi=300, xlabel= "", ylabel ="", xtickLabels = [], ytickLabels=[]):
    fig = plt.figure(figsize=figsize, layout="constrained")
    #plt.suptitle(title, fontsize=15)
    ax = sns.heatmap(colorValues, cmap = "Wistia", annot = textValues, annot_kws={"fontsize": 14})
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=12)
    ax.invert_yaxis()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if len(xtickLabels) == 0:
        plt.xticks(fontsize=14)
    else:
        plt.xticks(ticks=np.array(range(len(xtickLabels))) + 0.5, labels=xtickLabels, fontsize=12)
    if len(ytickLabels) == 0:
        plt.yticks(fontsize=14)
    else:
        plt.yticks(ticks=np.array(range(len(ytickLabels))) + 0.5, labels=ytickLabels,fontsize=12)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()

def saveGrid(matrix, title, filename, show=False, save = True, figsize=(4.2,3.8), dpi=300):
    fig, ax = plt.subplots(1,1,layout="constrained", figsize=figsize)
    #plt.suptitle(title,fontsize=15)
    maxVal = np.max(np.abs(matrix))
    cax = ax.imshow(matrix, cmap='bwr', vmin = -maxVal, vmax = maxVal)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(cax,shrink=0.9)
    cb.ax.tick_params(labelsize=14)
    if save:
        plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    else: 
        plt.close()