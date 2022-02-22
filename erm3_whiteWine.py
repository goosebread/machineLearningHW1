# Alex Yeh
# Question 3 White Wine Quality Dataset

from tkinter import Label
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

#labels from 0 to 10 are possible
#some labels are not seen in training data (only 3 to 9)
NLabels = 7
NFeatures = 11
LabelOffset = 3

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
    dataGivenClass = df[df['quality']==(i+LabelOffset)].loc[:, df.columns!='quality']
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
Decisions = torch.argmin(risks,dim=0)+LabelOffset

#calculate confusion matrix (using sklearn library)
#output matrix uses labels in sorted order (0 is label 3, 6 is label 9)
#ERROR TODO extend for all possible class labels
CM = torch.from_numpy(confusion_matrix(torch.flatten(trueClassLabels), torch.flatten(Decisions), normalize = 'true'))

Errors = torch.flatten(trueClassLabels)!=torch.flatten(Decisions)
pError = torch.sum(Errors).item()/NSamples #0.75 error...could be worse 0.85 error from blind guessing from 7 categories
print(pError)                               #drops to 0.5 error for alpha = 1e-5 (found using trial/error)

print(CM)

#PCA for graphing

#Do pca on the full sample set
#show the principle component dimensions that preserve most of the variance in the samples. 

#visualise: 7 to 11 shapes is too much to show clearly
#split into many graphs using color to showcase different aspects
#e.g. true class, predicted class, accuracy

#print(means)
#print(covs)
#m = torch.mean(t,dim=0)
#C = torch.cov(t.T)
#print(covs)

