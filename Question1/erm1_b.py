# Alex Yeh
# Question 1 Part B

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

N = 10000 #number of samples

#distribution of labels are known
#Covariance Matrix replaced with Diagonal Matrix preserving true variances
m0 = np.array([-1,-1,-1,-1])
C0 = np.array([[2,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,2]])
m1 = np.array([1,1,1,1])
C1 = np.array([[1,0,0,0],
            [0,2,0,0],
            [0,0,1,0],
            [0,0,0,3]])

mvn0 = multivariate_normal(m0,C0)
mvn1 = multivariate_normal(m1,C1)

samples = np.load(open('Q1Samples.npy', 'rb'))

pxgivenL0 = np.array([mvn0.pdf(samples)]).T
pxgivenL1 = np.array([mvn1.pdf(samples)]).T

ratio = pxgivenL1/pxgivenL0 #array of likelihood ratio for each sample

trueLabels = np.load(open('Q1Classes.npy', 'rb'))
totalPos = np.sum(trueLabels==1)
totalNeg = N - totalPos

#Generate ROC Curve
Ngammas = 8000
g1 = np.logspace(-20,-1,num=int(Ngammas/4),endpoint=False)
g2 = np.linspace(0.1,10,num=int(Ngammas/2),endpoint=False)
g3 = np.logspace(1,20,num=int(Ngammas/4))
gammas = np.concatenate((g1,g2,g3))

gammaResults = np.zeros((Ngammas,4))#col = {gamma, false negative, false positive, true positive}
gammaResults[:,0] = gammas

#future improvement: these could be done in parallel
for i in range(Ngammas):
    gamma = gammas[i]
    Decisions = ratio>gamma

    truePos = (Decisions==1) & (trueLabels==1)
    falsePos = (Decisions==1) & (trueLabels==0)
    ntp = np.sum(truePos)
    nfp = np.sum(falsePos)

    ptp = ntp/totalPos #probability of true positive 
    pfp = nfp/totalNeg #probability of false positive (type 1 error)

    falseNeg = (Decisions==0) & (trueLabels==1)
    pfn = np.sum(falseNeg)/totalPos #probability of false negative (type 2 error)

    gammaResults[i,1] = pfn #kept track of since section 2 specifically asks for it
    gammaResults[i,2] = pfp
    gammaResults[i,3] = ptp

#Section 3 - Calculate a (gamma,error) pair for min error
gammaErrors = (gammaResults[:,1] * totalPos + gammaResults[:,2] * totalNeg)/N
i_min = np.argmin(gammaErrors)

#Output select stats to console
print("N = "+str(N))
print("P(L=0) = "+str(totalNeg/N))
print("Minimum P(error) = "+str(gammaErrors[i_min]))
print("Gamma for min P(error) = "+str(gammas[i_min]))

#load ROC data from part A
ROCA = np.load(open('Q1_ROC1.npy', 'rb'))

# plot the ROC curve
fig,ax = plt.subplots()
#syntax note, the comma next to l0 helps the legend labels work
l0,=ax.plot(ROCA[:,0], ROCA[:,1], color='tab:green',zorder=0,label='True Covariance used')
l1,=ax.plot(gammaResults[:,2], gammaResults[:,3], color='tab:orange',zorder=1,label='Naive Bayes assumed')
l2=ax.scatter(gammaResults[i_min,2], gammaResults[i_min,3], color='tab:blue',marker='x',label='minimum P(error) for NB',zorder=2)
margin = 0.01
ax.set(xlim=(-margin, 1+margin), ylim=(-margin, 1+margin)) #display [0,1] on both axes

ax.set_xlabel('P(False Positive)')
ax.set_ylabel('P(True Positive)')
ax.set_title('Question 1 Part B ROC curve Approximation for NB Assumption')
ax.legend(handles=[l0,l1,l2])
ax.axis('equal')

plt.show()

#save ROC data for part C
with open('Q1_ROC2.npy', 'wb') as f1:
    np.save(f1, gammaResults[:,2:4])