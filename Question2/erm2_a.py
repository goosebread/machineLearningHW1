# Alex Yeh
# Question 2 Part A - 2,3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def runPartA(lossMatrix,title1,title2):

    #true data distribution is known for this ERM classifier
    distdata = np.load('Q2_DistData.npz')

    mvn1 = multivariate_normal(distdata['m1'],distdata['S1'])
    mvn2 = multivariate_normal(distdata['m2'],distdata['S2'])
    mvn3a = multivariate_normal(distdata['m3'],distdata['S3'])
    mvn3b = multivariate_normal(distdata['m4'],distdata['S4'])

    samples = np.load(open('Q2Samples.npy', 'rb'))

    pxgivenL1 = mvn1.pdf(samples)
    pxgivenL2 = mvn2.pdf(samples)
    #pdf of label 3 is a mixture with equal weights (0.5)
    pxgivenL3 = 0.5*mvn3a.pdf(samples)+0.5*mvn3b.pdf(samples)

    #class priors are given
    pL1 = 0.3
    pL2 = 0.3
    pL3 = 0.4
    #p(x) can be factored out as a constant 
    #and ignored since it doesn't affect the relative order of the computed risks
    pL1givenx = pxgivenL1 * pL1 # / p(x)
    pL2givenx = pxgivenL2 * pL2 # / p(x)
    pL3givenx = pxgivenL3 * pL3 # / p(x)

    #3xN matrix, each col represents probabilities for one sample
    P = np.stack((pL1givenx,pL2givenx,pL3givenx))

    #Loss matrix for minimum total error
    L = lossMatrix

    #Risk vectors for each sample, 3xN matrix
    R = np.matmul(L,P)
    
    #Make Decisions based on minimum risk
    Decisions = np.array([np.argmin(R, axis=0)])+1

    #Estimate minimum expected risk for using 10k samples
    minRisks = R.min(axis=0)
    expectedMinRisk = np.average(minRisks)
    print("Minimum Expected Risk = "+str(expectedMinRisk))

    #calculate confusion matrix (using sklearn library)
    trueLabels = np.load(open('Q2Classes.npy', 'rb')).T
    CM = confusion_matrix((trueLabels-1)[0], (Decisions-1)[0], normalize = 'true')
    print("Confusion Matrix: ")
    print(CM)

    #Part 3 Visualizations
    #separate by true label
    correctDecision = trueLabels==Decisions

    data = np.concatenate((samples,trueLabels.T,Decisions.T,correctDecision.T),axis=1)

    #this filtering scheme requires a reshape to return to 2d matrix representation
    data1 = data[np.argwhere(data[:,3]==1),:]
    data1 = np.reshape(data1,(data1.shape[0],data1.shape[2]))
    data2 = data[np.argwhere(data[:,3]==2),:]
    data2 = np.reshape(data2,(data2.shape[0],data2.shape[2]))
    data3 = data[np.argwhere(data[:,3]==3),:]
    data3 = np.reshape(data3,(data3.shape[0],data3.shape[2]))


    #plot Decisions vs Actual Label

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    l1=ax.scatter(data1[:,0],data1[:,1],zs=data1[:,2],marker = 'o',c = data1[:,4],label='Label = 1',cmap='Set1',vmin=1,vmax=3)
    l2=ax.scatter(data2[:,0],data2[:,1],zs=data2[:,2],marker = 's',c = data2[:,4],label='Label = 2',cmap='Set1',vmin=1,vmax=3)
    l3=ax.scatter(data3[:,0],data3[:,1],zs=data3[:,2],marker = 'v',c = data3[:,4],label='Label = 3',cmap='Set1',vmin=1,vmax=3)

    ax.set_title(title1)
    ax.legend(handles=[l1,l2,l3],title="Shape = True Label\nColor = ERM Decision")


    #plot Error vs Actual Label

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    cmap = LinearSegmentedColormap.from_list('redTransparentGreen', [(1, 0, 0, 1), (0.5, 1, 0.5, 0.1)])

    l12=ax2.scatter(data1[:,0],data1[:,1],zs=data1[:,2],marker = 'o',c = data1[:,5], label='Label = 1',cmap=cmap,vmin=0,vmax=1)
    l22=ax2.scatter(data2[:,0],data2[:,1],zs=data2[:,2],marker = 's',c = data2[:,5], label='Label = 2',cmap=cmap,vmin=0,vmax=1)
    l32=ax2.scatter(data3[:,0],data3[:,1],zs=data3[:,2],marker = 'v',c = data3[:,5], label='Label = 3',cmap=cmap,vmin=0,vmax=1)

    ax2.set_title(title2)
    lg = ax2.legend(handles=[l12,l22,l32],title="Red Marker = Incorrect Classification")
    for i in range(3):
        lg.legendHandles[i].set_alpha(1)
    plt.show()

