import numpy as np
import matplotlib.pyplot as plt

n=2 #dimensions

A = np.array([[0.9,-0.18],[0.56,0.29]])
#A = np.array([[0.2,0.5,0.3],[0.5,0.3,0.7],[0.1,0.2,0.3]])
#Sigma = np.matmul(A,A.T)

Sigma = np.array([[0.7,-0.5],[-0.5,0.8]])
print(Sigma)
#colvec = np.array([[0,1,1]]).T
#print(colvec)
#print(np.matmul(Sigma,colvec))

eig_vals,eig_vecs = np.linalg.eig(Sigma)
Q = eig_vecs#the columns are the eigenvectors!!!! not the rows!!!!
D = np.diag(eig_vals)
print(Q)
print(eig_vals)
print(Q[:,0])
print(D)

S2 = np.matmul(Q,np.matmul(D,Q.T))
print(S2)

"""
#only do this for positive semidefinate sigma
A1 = np.matmul(Q,np.sqrt(D))
print(A1)
print(np.matmul(A1,A1.T))
"""

#try cholesky decomposition for general case?
#also cries if input is not positive definite
A2 = np.linalg.cholesky(Sigma)
print(A2)
print(np.matmul(A2,A2.T))