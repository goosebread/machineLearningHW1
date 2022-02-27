# Alex Yeh
# Question 3 White Wine Quality Dataset
# helper function for selecting alpha hyperparameter

import torch
import numpy as np
from scipy.stats import multivariate_normal

#Assuming equal weighted losses in the loss matrix, the classification rule
#simplifies to Maximum aposteriori estimation
#A natural log can be applied to the MAP probability estimates to simplify calculations
#since it is monotonically increasing
def calcError(alpha, trueLabels, samples, Labels, NLabels, NFeatures):
    #cols are features
    #rows are class labels
    #WARNING some class labels have no data
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
    return pError