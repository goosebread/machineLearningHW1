#Alex Yeh
#Question 2 Sample Generator
#Part A-1

import numpy as np

#provided data: 

N = 10000 #number of Samples
n=3        #number of dimensions
p1 = 0.3 #class prior
p2 = 0.3 #class prior
p3 = 0.4 #class prior

#make 4 random cov matrices    
A1=np.random.rand(n,n)-0.5
S1=np.matmul(A1,A1.T) #ERROR all the covariances will be positive - we can randomly symmetrically make some negative later
A2=np.random.rand(n,n)-0.5
S2=np.matmul(A2,A2.T)
A3=np.random.rand(n,n)-0.5
S3=np.matmul(A3,A3.T)
A4=np.random.rand(n,n)-0.5
S4=np.matmul(A4,A4.T)

#get average standard deviation WARNING IS THIS METHOD VALID????
dist = 0
for S in [S1,S2,S3,S4]:
    dist += np.sum(np.sqrt(np.diag(S)))
dist = dist/12

#generate 4 means using a scaled tetrahedron-like shape

#3d tetrahedron sample coords on unit sphere from https://en.wikipedia.org/wiki/Tetrahedron
#new edge length is 2-3 times avg std deviation, old edge length is sqrt(8/9)
scale = (2+np.random.uniform()) * dist / np.sqrt(8/9) 
m1 = scale * np.array([np.sqrt(8/9),0,-1/3])#row vector
m2 = scale * np.array([-np.sqrt(2/9),np.sqrt(2/3),-1/3])
m3 = scale * np.array([-np.sqrt(2/9),-np.sqrt(2/3),-1/3])
m4 = scale * np.array([0,0,1])

#store mean/cov data to npz file
with open('Q2_DistData.npz', 'wb') as f0:
    np.savez(f0,m1=m1,m2=m2,m3=m3,m4=m4,S1=S1,S2=S2,S3=S3,S4=S4)

#generate true labels and samples
A = np.random.rand(N,1)
class1 = A<=p1 #0.3
class2 = (A<=p1+p2) & (A>p1) #0.3
class3a = (A<=p1+p2+p3/2) & (A>p1+p2) #0.2
class3b = A>p1+p2+p3/2 #0.2

trueClassLabels = class1 + 2*class2 + 3*class3a + 3*class3b
print("Class Priors")
print("p(L=1) = "+str(np.sum(trueClassLabels==1)/N))
print("p(L=1) = "+str(np.sum(trueClassLabels==2)/N))
print("p(L=1) = "+str(np.sum(trueClassLabels==3)/N))

x1 = np.random.multivariate_normal(m1, S1, N)
x2 = np.random.multivariate_normal(m2, S2, N)
x3a = np.random.multivariate_normal(m3, S3, N) 
x3b= np.random.multivariate_normal(m4, S4, N)

#class1,class2,class3 are mutually exclusive and collectively exhaustive
samples = class1*x1 + class2*x2 + class3a*x3a + class3b*x3b

#store true labels and samples
with open('Q2Classes.npy', 'wb') as f1:
    np.save(f1, trueClassLabels)

with open('Q2Samples.npy', 'wb') as f2:
    np.save(f2, samples)



