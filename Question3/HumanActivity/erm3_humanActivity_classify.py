# Alex Yeh
# Question 3 Human Activity Recognition

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

trueLabels=torch.load('trueLabels.pt')
samples=torch.load('samples.pt')
Labels = [1,2,3,4,5,6]

NLabels = len(Labels)
NFeatures = 561

alpha = 4.670e-6

means = torch.zeros([NLabels,NFeatures], dtype=torch.float64)
covs = torch.zeros([NLabels,NFeatures,NFeatures], dtype=torch.float64)
priors = torch.zeros([NLabels], dtype=torch.float64)

for i in range(NLabels):
    t = samples[trueLabels==Labels[i]]
    if torch.numel(t)==0:
        continue #the prior probability will be estimated as 0
    means[i] = torch.mean(t,dim=0)
    covt = torch.cov(t.T)
    #regularization term
    l = alpha * torch.trace(covt)/torch.linalg.matrix_rank(covt)
    covs[i] = covt + l*torch.eye(NFeatures)
    priors[i] = (t.size()[0])
NSamples = int(torch.sum(priors).item())
priors = priors/NSamples
lnClassPosteriors = torch.zeros([NLabels,NSamples], dtype=torch.float64)

#calculate class posteriors, ignoring common factor p(x)
for i in range(NLabels):
    if priors[i]==0:
        continue #skip if the prior probability is 0

    #multivariate gaussian. calling scipy function to give natural log of pdf value
    mvn = multivariate_normal(means[i],covs[i])
    pxGivenLabel =mvn.logpdf(samples)
    pLabelGivenx = pxGivenLabel + np.log(priors[i].item()) #/p(x) 
    lnClassPosteriors[i] = torch.from_numpy(pLabelGivenx)

#MAP classification rule
Decisions = torch.argmax(lnClassPosteriors,dim=0)+1
Errors = torch.flatten(trueLabels)!=torch.flatten(Decisions)
pError = torch.sum(Errors).item()/NSamples 
print("P(error) = "+str(pError))            

#calculate confusion matrix (using sklearn library)
#output matrix uses labels in sorted order
CM = pd.DataFrame(confusion_matrix(torch.flatten(trueLabels), torch.flatten(Decisions), labels = Labels, normalize = 'true'))
CM.to_csv("ConfusionMatrix_HumanActivity.csv",index=False) #save to file

#PCA for graphing

#Do pca on the full sample set
#show the principle component dimensions that preserve most of the variance in the samples. 

smean = torch.mean(samples,dim=0)
scov = torch.cov(samples.T)
xzm = samples - torch.matmul(smean.reshape(NFeatures,1),torch.ones((1,NSamples))).T
#get eigenvectors / principal component directions
eig_vals,eig_vecs = torch.linalg.eigh(scov)

#sort eigenvectors by eigenvalue magnitude
eig_vecs_sorted = eig_vecs[:,torch.argsort(eig_vals,descending = True)]
#eigenvectors are stored as columns

#apply transformation to align principal components with axes
y = torch.matmul((eig_vecs_sorted[:,(0,1,2)]).T,xzm.T).T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorlist = ['tab:purple','tab:blue','tab:cyan',
                'tab:green','tab:orange','tab:red']
cmap = LinearSegmentedColormap.from_list('cmap1', colorlist)

l1=ax.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = trueLabels/NLabels, cmap = cmap, alpha = 0.3)
ax.set_title("PCA (3 components) with true label colors for Human Activity")

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
l1b=ax2.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
ax2.set_title("PCA (3 components) with classifier decision colors for Human Activity")

#separate figure for marker legend
fig3,ax3 = plt.subplots()
labelNames = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
legend_elements = []
for l in range(NLabels):
    legend_elements.append(plt.scatter([0], [0], marker='o',alpha=0.3,color=colorlist[l], label=str(l+1)+' '+labelNames[l]))           
ax3.set_title("Label Colors")
ax3.legend(handles=legend_elements,loc='center',title="Label Colors",framealpha = 1)

plt.show()

