# Alex Yeh
# Question 3 White Wine Quality Dataset

import pandas as pd
import torch

#labels from 0 to 10 are possible
#some labels are not seen in training data (only 3 to 9)
NLabels = 7
NFeatures = 11
LabelOffset = 3

#hyperparameter alpha for regularization term
alpha = 0.1

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
    l = alpha * torch.trace(covt)/torch.matrix_rank(covt)
    covs[i] = covt + l*torch.eye(11)
    priors[i] = (t.size()[0])


#print(means)
#print(covs)
#m = torch.mean(t,dim=0)
#C = torch.cov(t.T)
#print(covs)


