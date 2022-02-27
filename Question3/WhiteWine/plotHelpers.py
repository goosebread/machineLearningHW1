import matplotlib.pyplot as plt

def plotSamples2D(samples,fnames,colorArray,cmap,NRows,NCols,title):
    fig, axs = plt.subplots(NRows, NCols)
    size = fnames.size
    for r in range(NRows):
        for c in range(NCols):
            f_index = (2*(r*NCols+c))
            axs[r,c].scatter(samples[:,f_index%size],samples[:,(f_index+1)%size],marker = 'o',c = colorArray, cmap = cmap, alpha = 0.3)
            axs[r,c].set_xlabel(fnames[f_index%size])
            axs[r,c].set_ylabel(fnames[(f_index+1)%size])
    fig.suptitle(title)
