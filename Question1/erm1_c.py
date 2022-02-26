# Alex Yeh
# Question 1 Part C

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

N = 10000 #number of samples

#Fisher LDA

#Separate Samples to estimate mean and cov for each label
samples = np.load(open('Q1Samples.npy', 'rb'))
trueLabels = np.load(open('Q1Classes.npy', 'rb'))
label1idx = np.argwhere(trueLabels==1)[:,0].T
label0idx = np.argwhere(trueLabels==0)[:,0].T

s0 = samples[label0idx,:]
s1 = samples[label1idx,:]

#estimates from sample, using column vectors
m0hat = np.array([np.mean(s0,axis=0)]).T
m1hat = np.array([np.mean(s1,axis=0)]).T
C0hat = np.cov(s0.T)
C1hat = np.cov(s1.T)
print("Label 0 mean vector estimate:")
print(m0hat)
print("Label 0 covariance matrix estimate:")
print(C0hat)
print("Label 1 mean vector estimate:")
print(m1hat)
print("Label 1 covariance matrix estimate:")
print(C1hat)

#Calculate Sb and Sw
Sb = np.matmul(m0hat-m1hat, (m0hat-m1hat).T)
Sw = C0hat + C1hat

#solve for LDA projection vector direction
eig_vals,eig_vecs = np.linalg.eig(np.matmul(np.linalg.inv(Sw),Sb))

#sort eigenvalues
eig_vecs_sorted = eig_vecs[:,np.argsort(-eig_vals)]
wLDA = eig_vecs_sorted[:,0]#row vector form/transposed already
print("Projection weight vector:")
print(wLDA)

totalPos = np.sum(trueLabels==1)
totalNeg = N - totalPos

#project data, wLDA is already a row vector
projSamples = np.array([np.matmul(wLDA,samples.T)]).T

#Generate ROC Curve
Ntaus = 8000
g1 = -np.logspace(20,1,num=int(Ntaus/4),endpoint=False)
g2 = np.linspace(-10,10,num=int(Ntaus/2),endpoint=False)
g3 = np.logspace(1,20,num=int(Ntaus/4))
taus = np.concatenate((g1,g2,g3))

#col = {tau, false negative, false positive, true positive}
tauResults = np.zeros((Ntaus,4))
tauResults[:,0] = taus

#future improvement: these could be done in parallel
for i in range(Ntaus):
    tau = taus[i]
    #Note that the "backwards" projection vector may be generated as a valid optimal solution
    #for that case, this comparison can be manually switched to a > operator to account for the extra 
    #negative sign
    Decisions = projSamples<tau

    truePos = (Decisions==1) & (trueLabels==1)
    falsePos = (Decisions==1) & (trueLabels==0)
    ntp = np.sum(truePos)
    nfp = np.sum(falsePos)

    ptp = ntp/totalPos #probability of true positive 
    pfp = nfp/totalNeg #probability of false positive (type 1 error)

    falseNeg = (Decisions==0) & (trueLabels==1)
    pfn = np.sum(falseNeg)/totalPos #probability of false negative (type 2 error)

    tauResults[i,1] = pfn #kept track of since section 2 specifically asks for it
    tauResults[i,2] = pfp
    tauResults[i,3] = ptp

#Section 3 - Calculate a (tau,error) pair for min error
tauErrors = (tauResults[:,1] * totalPos + tauResults[:,2] * totalNeg)/N
i_min = np.argmin(tauErrors)

#Output select stats to console
print("N = "+str(N))
print("P(L=0) = "+str(totalNeg/N))
print("Minimum P(error) = "+str(tauErrors[i_min]))
print("Tau for min P(error) = "+str(taus[i_min]))

#load ROC data from part A,B
ROCA = np.load(open('Q1_ROC1.npy', 'rb'))
ROCB = np.load(open('Q1_ROC2.npy', 'rb'))

# plot the ROC curve
fig,ax = plt.subplots()
l0,=ax.plot(ROCA[:,0], ROCA[:,1], color='tab:green',zorder=0,label='True Covariance used')
l1,=ax.plot(ROCB[:,0], ROCB[:,1], color='tab:pink',zorder=1,label='Naive Bayes assumed')
l2,=ax.plot(tauResults[:,2], tauResults[:,3], color='tab:orange',zorder=2,label='Fisher LDA')
l3=ax.scatter(tauResults[i_min,2], tauResults[i_min,3], color='tab:blue',marker='x',label='minimum P(error)',zorder=3)
margin = 0.01
ax.set(xlim=(-margin, 1+margin), ylim=(-margin, 1+margin)) #display [0,1] on both axes

ax.set_xlabel('P(False Positive)')
ax.set_ylabel('P(True Positive)')
ax.set_title('Question 1 Part C ROC curve Approximation for Fisher LDA')
ax.legend(handles=[l0,l1,l2,l3])
ax.axis('equal')

plt.show()

