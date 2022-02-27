# Alex Yeh
# Question 3 Human Activity Recognition
# program for selecting alpha hyperparameter

# This runs the classifier for many values of alpha, records the min error,
# and outputs the alpha that gives the lowest min error

# This brute-force approach is simple to implement 
# but computationally expensive/not optimal

import pandas as pd
import numpy as np
from calcError import *
import matplotlib.pyplot as plt

trueLabels=torch.load('trueLabels.pt')
samples=torch.load('samples.pt')
Labels = [1,2,3,4,5,6]

NLabels = len(Labels)
NFeatures = 561

Nalphas=500
alphas = np.logspace(0,-10,Nalphas)
minErrors = np.ones(Nalphas)

#hyperparameter alpha for regularization term
for i in range(Nalphas):
    try:
        minErrors[i]=calcError(alphas[i],trueLabels,samples,Labels,NLabels,NFeatures)
    except:
        continue

i_min = np.argmin(minErrors)
print("Optimal Alpha: ")
print(alphas[i_min])

# plot alpha vs min error
fig,ax = plt.subplots()
l0,=ax.semilogx(alphas, minErrors, color='tab:orange',zorder=0)
l3=ax.scatter(alphas[i_min], minErrors[i_min], color='tab:blue',marker='x',zorder=1)

ax.set_xlabel('alpha')
ax.set_ylabel('Min Error')
ax.set_title('Empirically selecting hyperparameter alpha')

plt.show()
