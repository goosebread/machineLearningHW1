#attempt at doing PCA algorithm in python
#cuz matlab student license will run out one day

import numpy as np
import matplotlib.pyplot as plt

n=2 #dimensions
N=100 #samples 
mu = [[100],[10]]#100 * np.ones((n,1)) #mu = [100, 100]'

A = np.random.rand(n,n)
Sigma = np.matmul(A,A.T) #generated covariance

#samples
#note: we can still do pca if for some reason covariance is not positive definate. we just can't generate samples. really hope that situation doesn't happen lol
x = np.matmul(A,np.random.normal(0,1,(n,N))) + np.matmul(mu,np.ones((1,N)))

#estimate from samples
muhat = np.array([np.mean(x,axis=1)]).T#to make it a column vec
Sigmahat = np.cov(x)

print(A)
print(Sigma)
print(Sigmahat)

#convert to zero mean samples
xzm = x - np.matmul(muhat,np.ones((1,N)))
#get eigenvectors / principal component directions
eig_vals,eig_vecs = np.linalg.eig(Sigmahat)

#sort eigenvectors by eigenvalue magnitude
eig_vecs_sorted = eig_vecs[:,np.argsort(-eig_vals)]
#argsort sorts from smallest to largest. negative sign is a cheeky workaround

#apply transformation to align principal components with axes
y = np.matmul((eig_vecs_sorted).T,xzm)

fig=plt.figure(1)
ax1 = fig.add_subplot(2,2,1)
ax1.scatter(x[0,:],x[1,:])
ax1.axis('equal')
ax2 = fig.add_subplot(2,2,2)
ax2.scatter(xzm[0,:],xzm[1,:])
ax2.axis('equal')

ax = fig.add_subplot(2,2,3)
ax.scatter(y[0,:],y[1,:])
ax.set(xlim=([-2, 2]),ylim=([-2, 2]))
ax.axis('equal')

plt.show()