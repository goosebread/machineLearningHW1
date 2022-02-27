# Alex Yeh
# Question 3 White Wine Quality Dataset

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from plotHelpers import *

NLabels = 11
Labels = range(NLabels) #list of all labels
NFeatures = 11

#hyperparameter alpha for regularization term
alpha = 2.575e-5

df = pd.read_csv('data/winequality-white.csv',sep=';')
FeatureNames = df.columns[:-1]

trueClassLabels = torch.tensor(df.loc[:,df.columns=='quality'].values)
samples = torch.tensor(df.loc[:,df.columns!='quality'].values)

#cols are features
#rows are class labels
#WARNING some class labels have no data
means = torch.zeros([NLabels,NFeatures], dtype=torch.float64)
covs = torch.zeros([NLabels,NFeatures,NFeatures], dtype=torch.float64)
priors = torch.zeros([NLabels], dtype=torch.float64)

for i in range(NLabels):
    dataGivenClass = df[df['quality']==i].loc[:, df.columns!='quality']
    if dataGivenClass.empty:
        continue #the prior probability will be estimated as 0
    t = torch.tensor(dataGivenClass.values)
    means[i] = torch.mean(t,dim=0)
    covt = torch.cov(t.T)
    #regularization term
    l = alpha * torch.trace(covt)/torch.linalg.matrix_rank(covt)
    covs[i] = covt + l*torch.eye(11)
    priors[i] = (t.size()[0])
NSamples = int(torch.sum(priors).item())
priors = priors/NSamples

classPosteriors = torch.zeros([NLabels,NSamples], dtype=torch.float64)

#calculate class posteriors, ignoring common factor p(x)
for i in range(NLabels):
    if priors[i]==0:
        continue #skip if the prior probability is 0
    mvn = multivariate_normal(means[i],covs[i])
    pxGivenLabel =mvn.pdf(samples)
    pLabelGivenx = pxGivenLabel * priors[i].item() #/p(x) 
    classPosteriors[i] = torch.from_numpy(pLabelGivenx)

#loss matrix, minimize total error
L = torch.ones([NLabels,NLabels], dtype=torch.float64) - torch.eye(NLabels)

#calculate Risk Matrix for all samples (NDecisions x NSamples)
#(NDecisions = NLabels)
risks = torch.matmul(L,classPosteriors)
Decisions = torch.argmin(risks,dim=0)

#calculate confusion matrix (using sklearn library)
#output matrix uses labels in sorted order
CM = pd.DataFrame(confusion_matrix(torch.flatten(trueClassLabels), torch.flatten(Decisions), labels = Labels, normalize = 'true'))
CM.to_csv("ConfusionMatrix_WhiteWine.csv",index=False) #save to file

Errors = torch.flatten(trueClassLabels)!=torch.flatten(Decisions)
pError = torch.sum(Errors).item()/NSamples 
print("P(error) = "+str(pError))            

#PCA for graphing

#Do pca on the full sample set
#show the principle component dimensions that preserve most of the variance in the samples. 

smean = torch.mean(samples,dim=0)
scov = torch.cov(samples.T)
xzm = samples - torch.matmul(smean.reshape(NLabels,1),torch.ones((1,NSamples),dtype = torch.double)).T
#get eigenvectors / principal component directions
eig_vals,eig_vecs = torch.linalg.eigh(scov)

#sort eigenvectors by eigenvalue magnitude
eig_vecs_sorted = eig_vecs[:,torch.argsort(eig_vals,descending = True)]
#eigenvectors are stored as columns

#apply transformation to align principal components with axes
y = torch.matmul((eig_vecs_sorted[:,(0,1,2)]).T,xzm.T).T

#visualise: 7 to 11 shapes is too much to show clearly
#split into many graphs using color to showcase different aspects

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorlist = ['k','tab:brown','tab:olive','tab:purple','tab:blue','tab:cyan',
                'tab:green','tab:orange','tab:red','tab:pink','tab:gray']
cmap = LinearSegmentedColormap.from_list('cmap1', colorlist)

colorLabels = ['k','tab:brown','tab:olive','tab:purple','tab:blue','tab:cyan',
                'tab:green','tab:orange','tab:red','tab:pink','tab:gray']
l1=ax.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)#,c = data1[:,4],label='Label = 1',cmap='Set1',vmin=1,vmax=3)
ax.set_title("PCA (3 components) with true label colors")

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
l1b=ax2.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)#,c = data1[:,4],label='Label = 1',cmap='Set1',vmin=1,vmax=3)
ax2.set_title("PCA (3 components) with classifier decision colors")

#separate figure for marker legend
fig3,ax3 = plt.subplots()
legend_elements = []
for l in range(NLabels):
    legend_elements.append(plt.scatter([0], [0], marker='o',alpha=0.3,color=colorlist[l], label='Label '+str(l)))           
ax3.set_title("Label Colors")
ax3.legend(handles=legend_elements,loc='center',title="Label Colors",framealpha = 1)


#6 2d graphs in subplots to show full data along feature axes
plotSamples2D(samples,FeatureNames,trueClassLabels/NLabels,cmap,2,3,"True Class Labels")
plotSamples2D(samples,FeatureNames,Decisions/NLabels,cmap,2,3,"Classifier Decisions")
plt.show()

