
from re import S
import numpy as np
import matplotlib.pyplot as plt

n=2
N1=100
N2=75

#true params for synthetic data
mu1=-np.ones((n,1))
A1=3*np.random.rand(n,n)-0.5 #true cov = A*A.T
#mu2=6*np.ones((n,1))
mu2 = np.array([[-1],[5]])
A2 = 2*np.random.rand(n,n)-0.5

#generate samples
x1 = (np.matmul(A1,np.random.normal(0,1,(n,N1))) 
            + np.matmul(mu1,np.ones((1,N1))))
x2 = (np.matmul(A2,np.random.normal(0,1,(n,N2))) 
            + np.matmul(mu2,np.ones((1,N2))))

#measure/estimate stats
mu1hat = np.array([np.mean(x1,axis=1)]).T
print(mu1hat)
mu2hat = np.array([np.mean(x2,axis=1)]).T
S1hat = np.cov(x1)
S2hat = np.cov(x2)

#Calculate Sb and Sw
Sb = np.matmul(mu1hat-mu2hat, (mu1hat-mu2hat).T)
Sw = S1hat + S2hat

#solve for LDA projection vector direction
eig_vals,eig_vecs = np.linalg.eig(np.matmul(np.linalg.inv(Sw),Sb))

#sort eigenvalues
eig_vecs_sorted = eig_vecs[:,np.argsort(-eig_vals)]
wt = eig_vecs_sorted[:,0]#row vector form/transposed already
print(wt)
#project data
y1 = np.matmul(wt,x1)
y2 = np.matmul(wt,x2)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(x1[0,:], x1[1,:], color='tab:blue')
ax.scatter(x2[0,:], x2[1,:], color='tab:orange')

ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(y1, np.zeros(N1), color='tab:blue')
ax2.scatter(y2, np.zeros(N2), color='tab:orange')

#ax.set(xlim=(, 8), ylim=(0, 8))
plt.show()


