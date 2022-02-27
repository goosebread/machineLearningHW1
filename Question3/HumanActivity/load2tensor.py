import torch
import numpy as np

NSamples = 7352 #7352/10299 total samples are in the training set
NFeatures = 561

#load samples
file1 = open('UCI HAR Dataset/train/X_train.txt', 'r')
lines = file1.readlines()
samples = torch.zeros((NSamples,NFeatures))
for i in range(len(lines)):
    samples[i,:]=torch.from_numpy(np.float_(lines[i].split()))
print("Sample Data Dimensions: ")
print(samples.shape)
file1.close()
torch.save(samples, 'samples.pt')

#load true class labels
file2 = open('UCI HAR Dataset/train/y_train.txt', 'r')
lines = file2.readlines()
trueLabels = torch.zeros((NSamples))
for i in range(len(lines)):
    trueLabels[i]=float(lines[i])
file2.close()
print("True Label Dimensions: ")
print(trueLabels.shape)
torch.save(trueLabels, 'trueLabels.pt')
