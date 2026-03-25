from matplotlib import pyplot as plt
import numpy as np


# change some default parameters for easier diagrams
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.dpi": 300,
})

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
                  yAxisLinearLim = 1, quantileLower = [], quantileHigher = [], xTickLabels = True, yTickLabels = True, fontsizeFactor = 1,
                  xLims = None, yLims = None, moveYLabel = 0, greyAxisAt = None, closePlot = True, fig = None,
                  alpha_fillBetween = 0.2):
    """Draws a plot with lines for data rows, optionally with error bars.
    \nScores are the MCC score values, in a 2D array with first dimension as the number of data lines drawn, \
    and second dimension equaling values dimension. 
    \nValues are drawn on the x-axis, e.g. number of samples"""
    if fig is None:
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
    
    plt.xlabel(xlabel, fontsize=14*fontsizeFactor)
    plt.ylabel(ylabel, fontsize=14*fontsizeFactor)
    
    plt.xticks(fontsize=14*fontsizeFactor)
    plt.yticks(fontsize=14*fontsizeFactor)
    ax = plt.gca()
    if not xTickLabels:
        ax.set(xticklabels=[])
    if not yTickLabels:
        ax.set(yticklabels=[])
    if xLims:
        ax.set_xlim(xLims[0], xLims[1])
    if yLims:
        ax.set_ylim(yLims[0], yLims[1])
    if moveYLabel != 0:
        import matplotlib.transforms as mtransforms
        # x0, y0 = ax.yaxis.get_label().get_position()
        offset = mtransforms.ScaledTranslation(0, moveYLabel/72, ax.figure.dpi_scale_trans)
        # ax.yaxis.set_label_coords(x0, y0 + moveYLabel)
        ax.yaxis.get_label().set_transform(ax.yaxis.get_label().get_transform() + offset)
    if greyAxisAt is not None:
        plt.axvline(greyAxisAt, color ="grey")
    if len(scores.shape) == 1:
        plt.plot(values, scores)
        if len(errors) > 0:
            plt.fill_between(values, scores + errors, scores - errors, alpha = alpha_fillBetween)
        elif len(quantileLower) > 0:
            plt.fill_between(values, quantileLower, quantileHigher, alpha = alpha_fillBetween)
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
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = alpha_fillBetween)
                elif len(quantileLower) > 0:
                    plt.fill_between(values, quantileLower[i], quantileHigher[i], alpha = alpha_fillBetween)
            else:
                plt.plot(values, scores[i], label = str(rowLabels[i]), color=colors[i])
                if len(errors) > 0:
                    plt.fill_between(values, scores[i] + errors[i], scores[i] - errors[i], alpha = alpha_fillBetween, color= colors[i])
                elif len(quantileLower) > 0:
                    plt.fill_between(values, quantileLower[i], quantileHigher[i], alpha = alpha_fillBetween, color= colors[i])
        if showLegend:
            order = [2, 0, 1]
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], fontsize=14*fontsizeFactor)
    if save:
        plt.savefig(filename, dpi=dpi)
        # quick and dirty: if filename is not specified (up to 4 letters), then save a pdf additionally to the png above.
        if len(str(filename).split('.')[-1]) > 4:
            plt.savefig(filename+".pdf", dpi=dpi)
    if show:
        plt.show()
    elif closePlot:
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
            mayShowLegend = True
            if len(rowLabels) == 0:
                rowLabelTuple = np.zeros(scoreTuple.shape[0])
                mayShowLegend = False
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
            if mayShowLegend:
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
                  yAxisLinearLim = 1, quantileLower = [], quantileHigher = [], marker = 'o',
                   linestyles = [], legend_outside = False, ylim = None, legend = True, xTickLabels = True, yTickLabels=True, fontsizeFactor = 1,
                   legendColumns = 1):
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
    plt.xlabel(xlabel, fontsize=14*fontsizeFactor)
    plt.ylabel(ylabel, fontsize=14*fontsizeFactor)
    plt.xticks(fontsize=14*fontsizeFactor)
    plt.yticks(fontsize=14*fontsizeFactor)
    ax = plt.gca()
    if not xTickLabels:
        ax.set(xticklabels=[])
    if not yTickLabels:
        ax.set(yticklabels=[])
    if len(scores.shape) == 1:
        if len(errors) > 0:
            plt.errorbar(values, scores, errors, fmt=marker, markersize=10, alpha = 1, elinewidth=3, capsize=6)
        elif len(quantileLower) > 0:
            plt.errorbar(values, scores, np.stack((quantileLower, quantileHigher), axis=0), fmt=marker, alpha = 1, elinewidth=3, capsize=6)
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
                    plt.errorbar(values, scores[i], errors[i], fmt=marker, markersize=15, alpha = 1,label = str(rowLabels[i]), elinewidth=3, capsize=6)
                elif len(quantileLower) > 0:
                    plt.errorbar(values, scores[i], np.stack((quantileLower[i], quantileHigher[i]), axis=0), fmt=marker, alpha = 1,label = str(rowLabels[i]), elinewidth=3, capsize=6)
                else:
                    plt.scatter(values, scores[i],  marker=marker,label = str(rowLabels[i]))
            else:
                if len(errors) > 0:
                    plt.errorbar(values, scores[i],errors[i], fmt="none", markersize=15, alpha = 1, color= colors[i], elinewidth=3, capsize=6)
                    plt.errorbar(values, scores[i], fmt=marker, markersize=10, alpha = 1.0, color= colors[i], elinewidth=3, capsize=6)
                    if len(linestyles) > 0:
                        plt.plot(values, scores[i], color = colors[i],label = str(rowLabels[i]), linestyle=linestyles[i])
                    else:
                        plt.plot(values, scores[i], color = colors[i],label = str(rowLabels[i]))
                elif len(quantileLower) > 0:
                    plt.errorbar(values, scores[i], np.stack((quantileLower[i], quantileHigher[i]), axis=0), fmt=marker, alpha = 1, color= colors[i],label = str(rowLabels[i]), elinewidth=3, capsize=6)
                else:
                    plt.scatter(values, scores[i], label = str(rowLabels[i]), marker=marker, color=colors[i])
        if showLegend and legend:
            if legend_outside:
                plt.legend(ncol=legendColumns,fontsize=14*fontsizeFactor, bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                plt.legend(ncol=legendColumns,fontsize=14*fontsizeFactor)
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

def pyGraphVizCouplingMatrix(matrix, filename, dpi = 300):
    import pygraphviz as pgv
    import networkx as nx

    n = matrix.shape[0]
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    A = pgv.AGraph(directed=True, strict=False)
    for i in range(n):
        A.add_node(i, shape='circle', style='filled', fillcolor='orange', width=0.6)
    for u,v, d in G.edges(data=True):
        color = 'blue' if d['weight'] == 1 else 'red'
        A.add_edge(u, v, color=color, penwidth=3, arrowsize=1.2)

    for node in A.nodes():
        node.attr['width'] = '0.6'
        node.attr['height'] = '0.6'
    A.graph_attr['K'] = '0.5'
    A.graph_attr['size'] = '4.5,4.5'   # inches, equivalent to figsize=(4.5, 4.5)
    A.graph_attr['dpi'] = str(dpi)
    A.graph_attr['splines'] = 'true'
    A.graph_attr['overlap'] = 'false'
    A.layout(prog='fdp')   # neato gives circular-ish organic layouts
    A.draw(filename)

def customCouplingMatrixGraph(matrix, title, filename, show = False, save= True, figsize=(4.5,4.5), dpi=300):
    import networkx as nx
    edge_colors = {1: 'tab:blue', -1: 'tab:red'}

    G = nx.DiGraph()

    n = matrix.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if matrix[i, j] != 0:
                G.add_edge(i, j, weight=matrix[i, j])

    pos = nx.circular_layout(G, scale=0.8)
    print(pos)
    edge_color_list = [edge_colors[G[u][v]['weight']] for u, v in G.edges()]
        
    fig, ax = plt.subplots(1,1, figsize=figsize, layout="constrained")
    # plt.title(title, fontsize=17)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color = edge_color_list, connectionstyle=f'arc3, rad={0.2}', arrows=True, width=5, arrowsize=30, node_size=1600)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='orange', node_size=1600)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_color='black')  

    # # --- Force equal, symmetric limits ---
    # xvals, yvals = np.array(list(pos.values())).T
    # pad = 0.1
    # xmax, ymax = xvals.max() + pad, yvals.max() + pad
    # xmin, ymin = xvals.min() - pad, yvals.min() - pad
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_aspect('equal')

    # # Remove frame/padding
    # plt.tight_layout()
    ax.margins(0.1)
    ax.axis('off')
    

    # --- Save cleanly ---
   
    if save:
        plt.savefig(filename, dpi=dpi, pad_inches=0)
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

def plotCouplingGraphs():
    matrices = [defaultCouplingMatrixCascade_LowDense, defaultCouplingMatrixCascade_HighDense,
                mediumCouplingMatrixCascade_LowDense, mediumCouplingMatrixCascade_HighDense,
                largeCouplingMatrixCascade_LowDense, largeCouplingMatrixCascade_HighDense]
    filenames = ["SmallLowDense", "SmallHighDense", "MedLowDense", "MedHighDense", "LargeLowDense", "LargeHighDense"]
    for matr, name in zip(matrices, filenames):
        pyGraphVizCouplingMatrix(matr, "diagrams/MatrixGraphs/" +name + ".png")
        # customCouplingMatrixGraph(matr, "", "diagrams/MatrixGraphs/" +name + ".png")

def plotRandomCouplingGraphs():
    from PIL import Image
    matrices = np.load("data/random_graphs.npy")
    matrices = matrices[1:]
    for i, matr in enumerate(matrices):
        pyGraphVizCouplingMatrix(matr, "diagrams/MatrixGraphs/" +str(i) + ".png")
        # customCouplingMatrixGraph(matr, "", "diagrams/MatrixGraphs/" +name + ".png")
    fig, axes = plt.subplots(2,5, figsize=(6,3))
    for i, ax in enumerate(axes.flatten()):
        img = Image.open("diagrams/MatrixGraphs/"+str(i)+".png")
        ax.imshow(img)
        ax.axis("off")
    labels = []
    texts = "abcdefghik"
    for letter in texts:
        labels.append('('+letter+')')

    for ax, label in zip(axes.flatten(), labels):
        ax.text(
            -0.1, 0.98, label,
            transform=ax.transAxes,
            fontsize=7,
            va='top',
            ha='left'
        )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.01)
    # fig.tight_layout()
    plt.savefig("diagrams/MatrixGraphs/combination.png", dpi=300)

if __name__ == "__main__":
    plotRandomCouplingGraphs()