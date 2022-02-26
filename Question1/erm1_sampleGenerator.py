#Alex Yeh
#Question 1 Sample Generator

import numpy as np

#provided data: 
N = 10000 #number of Samples
p0 = 0.7 #class prior

#using row vectors
m0 = np.array([-1,-1,-1,-1])
C0 = np.array([[2,-0.5,0.3,0],
            [-0.5,1,-0.5,0],
            [0.3,-0.5,1,0],
            [0,0,0,2]])

m1 = np.array([1,1,1,1])
C1 = np.array([[1,0.3,-0.2,0],
            [0.3,2,0.3,0],
            [-0.2,0.3,1,0],
            [0,0,0,3]])

#generate true class label order
A = np.random.rand(N,1)
trueClassLabels = A>p0 # True/1 for Class 1, False/0 for Class 0

#This generates double the data compared to a for-loop scheme but is more parallel
x0 = np.random.multivariate_normal(m0, C0, N)
x1 = np.random.multivariate_normal(m1, C1, N)

#each row is 1 sample taken from x1 if Label1, x0 otherwise
samples = np.where(trueClassLabels,x1,x0)

#save data set
with open('Q1Classes.npy', 'wb') as f1:
    np.save(f1, trueClassLabels)

with open('Q1Samples.npy', 'wb') as f2:
    np.save(f2, samples)

#Consistency checks 
"""
numClass1 = np.sum(trueClassLabels)
print(numClass1/N)# consistency check - P(L = 1) should be around 0.3

#manual check for small N
print(trueClassLabels) # check that samples and true class labels match
print(x1)
print(samples)
"""