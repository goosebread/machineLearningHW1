#Alex Yeh
#Python script for graphing values from csv file

import pandas as pd
import matplotlib.pyplot as plt

filename = "circlePoints.csv"
data = pd.read_csv(filename)

seeds = data["seed"].unique()
radii = data["radius"].unique()

#separate by seed and radius
#there should be 3 seeds and 3 radii
data11 = data.loc[(data["seed"] == seeds[0])].loc[(data["radius"] == radii[0])]
data12 = data.loc[(data["seed"] == seeds[0])].loc[(data["radius"] == radii[1])]
data13 = data.loc[(data["seed"] == seeds[0])].loc[(data["radius"] == radii[2])]

data21 = data.loc[(data["seed"] == seeds[1])].loc[(data["radius"] == radii[0])]
data22 = data.loc[(data["seed"] == seeds[1])].loc[(data["radius"] == radii[1])]
data23 = data.loc[(data["seed"] == seeds[1])].loc[(data["radius"] == radii[2])]

data31 = data.loc[(data["seed"] == seeds[2])].loc[(data["radius"] == radii[0])]
data32 = data.loc[(data["seed"] == seeds[2])].loc[(data["radius"] == radii[1])]
data33 = data.loc[(data["seed"] == seeds[2])].loc[(data["radius"] == radii[2])]

#color for seed
#number x, p y
#use semilog

# plot the data
fig,ax = plt.subplots()
l11=ax.scatter(data11["N"], data11["p"], color='tab:blue', alpha = 0.3, marker = 'o', label = (str(seeds[0])+","+str(radii[0])))
l12=ax.scatter(data12["N"], data12["p"], color='tab:blue', alpha = 0.3, marker = 's', label = (str(seeds[0])+","+str(radii[1])))
l13=ax.scatter(data13["N"], data13["p"], color='tab:blue', alpha = 0.3, marker = '*', label = (str(seeds[0])+","+str(radii[2])))

l21=ax.scatter(data21["N"], data21["p"], color='tab:orange', alpha = 0.3, marker = 'o', label = (str(seeds[1])+","+str(radii[0])))
l22=ax.scatter(data22["N"], data22["p"], color='tab:orange', alpha = 0.3, marker = 's', label = (str(seeds[1])+","+str(radii[1])))
l23=ax.scatter(data23["N"], data23["p"], color='tab:orange', alpha = 0.3, marker = '*', label = (str(seeds[1])+","+str(radii[2])))

l31=ax.scatter(data31["N"], data31["p"], color='tab:red', alpha = 0.3, marker = 'o', label = (str(seeds[2])+","+str(radii[0])))
l32=ax.scatter(data32["N"], data32["p"], color='tab:red', alpha = 0.3, marker = 's', label = (str(seeds[2])+","+str(radii[1])))
l33=ax.scatter(data33["N"], data33["p"], color='tab:red', alpha = 0.3, marker = '*', label = (str(seeds[2])+","+str(radii[2])))

ax.set_xscale('log')

ax.set_xlabel('N (Trial Count)')
ax.set_ylabel('Probability')
ax.set_title('Part 4 Simulation Results')
ax.legend(title = "Seed, Radius")

#ax.set(xlim=(, 8), ylim=(0, 8))
plt.show()
