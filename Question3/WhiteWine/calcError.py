# Alex Yeh
# Question 3 White Wine Quality Dataset
# helper function for selecting alpha hyperparameter

import torch
import pandas as pd
from scipy.stats import multivariate_normal

def calcError(alpha, df, NLabels, NFeatures):
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
    Errors = torch.flatten(trueClassLabels)!=torch.flatten(Decisions)
    pError = torch.sum(Errors).item()/NSamples 
    return pError