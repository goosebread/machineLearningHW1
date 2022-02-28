This is the code for HW1 of EECE5644.
The code was written by Alex Yeh and functions from the following libraries were used:
numpy, pandas, torch, sklearn, skipy, matplotlib

All scripts should be called at the Question-folder level.

To run the code from Question 1 or Question 2, first run the sampleGenerator script to create the sample files.
The scripts from Question 1 should be run in the order erm1_a, erm1_b, erm1_c so the ROC curves can be compared properly.
The scripts from Question 2 should be run by calling erm2_b. 

For the White Wine part of Question 3, erm3_whiteWine_chooseAlpha.py can be run to return an optimal alpha hyperparameter value.
erm3_whiteWine_classify.py generates the graphs and confusion matrix for the data set, with the optimal alpha hard-coded.

For the Human Activity part of Question 3, load2tensor.py must first be run to extract the data from the raw txt file and save it as the .pt pytorch tensor files. erm3_humanActivity_chooseAlpha.py can be run to return an optimal alpha hyperparameter value.
erm3_humanActivity_classify.py generates the graphs and confusion matrix for the data set, with the optimal alpha hard-coded.

Note: there are some slight differences between the calcError.py functions of the wine and human activity problems. The wine version makes decisions based on the class conditional pdf and the prior, while the human activity version uses the natural log of these values. However, these differences do not affect the end results since they both still follow the MAP classification rule. 