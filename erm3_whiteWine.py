# Alex Yeh
# Question 3 White Wine Quality Dataset

import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

#labels from 0 to 10 are possible
#some labels are not seen in training data (only 3 to 9)
NLabels = 11
Labels = range(NLabels) #list of all labels
NFeatures = 11

#hyperparameter alpha for regularization term
alpha = 1e-5

df = pd.read_csv('data/winequality-white.csv',sep=';')
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
#could propagate nan values all the way here(for labels that never show up) and just make sure argmin never picks them
Decisions = torch.argmin(risks,dim=0)

#calculate confusion matrix (using sklearn library)
#output matrix uses labels in sorted order (0 is label 3, 6 is label 9)
#ERROR TODO extend for all possible class labels
CM = torch.from_numpy(confusion_matrix(torch.flatten(trueClassLabels), torch.flatten(Decisions), labels = Labels, normalize = 'true'))

Errors = torch.flatten(trueClassLabels)!=torch.flatten(Decisions)
pError = torch.sum(Errors).item()/NSamples #0.75 error...could be worse 0.85 error from blind guessing from 7 categories
print(pError)                               #drops to 0.5 error for alpha = 1e-5 (found using trial/error)

print(CM)

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
print(y.shape)

#visualise: 7 to 11 shapes is too much to show clearly
#split into many graphs using color to showcase different aspects
#e.g. true class, predicted class, accuracy

#true class visualization

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
cmap = LinearSegmentedColormap.from_list('cmap1', ['k','tab:brown','tab:olive','tab:purple','tab:blue','tab:cyan',
                'tab:green','tab:orange','tab:red','tab:pink','tab:gray'])

colorLabels = ['k','tab:brown','tab:olive','tab:purple','tab:blue','tab:cyan',
                'tab:green','tab:orange','tab:red','tab:pink','tab:gray']
l1=ax.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)#,c = data1[:,4],label='Label = 1',cmap='Set1',vmin=1,vmax=3)


fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
l1b=ax2.scatter(y[:,0],y[:,1],zs=y[:,2],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)#,c = data1[:,4],label='Label = 1',cmap='Set1',vmin=1,vmax=3)

plt.show()
"""
legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
                   Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='g', markersize=15),
                   Patch(facecolor='orange', edgecolor='r',
                         label='Color Patch')]
                         """
#ax.set_title(title1)
#ax.legend(handles=legend_elements,title="Shape = True Label\nColor = ERM Decision")


#maybe its better to have like 6 2d graphs in subplots to show full data along feature axes
fig3, axs = plt.subplots(3, 2)
l0t=axs[0,0].scatter(samples[:,0],samples[:,1],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)
l1t=axs[0,1].scatter(samples[:,2],samples[:,3],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)
l2t=axs[1,0].scatter(samples[:,4],samples[:,5],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)
l3t=axs[1,1].scatter(samples[:,6],samples[:,7],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)
l4t=axs[2,0].scatter(samples[:,8],samples[:,9],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)
l5t=axs[2,1].scatter(samples[:,10],samples[:,0],marker = 'o',c = trueClassLabels/NLabels, cmap = cmap, alpha = 0.3)

fig4, axs2 = plt.subplots(3, 2)
l0d=axs2[0,0].scatter(samples[:,0],samples[:,1],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
l1d=axs2[0,1].scatter(samples[:,2],samples[:,3],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
l2d=axs2[1,0].scatter(samples[:,4],samples[:,5],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
l3d=axs2[1,1].scatter(samples[:,6],samples[:,7],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
l4d=axs2[2,0].scatter(samples[:,8],samples[:,9],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
l5d=axs2[2,1].scatter(samples[:,10],samples[:,0],marker = 'o',c = Decisions/NLabels, cmap = cmap, alpha = 0.3)
plt.show()