#Alex Yeh
#Python script for graphing values from csv file

import pandas as pd
import matplotlib.pyplot as plt

filename = "loadedDice.csv"
data = pd.read_csv(filename)

seeds = data["seed"].unique()

#separate by seed
#there should be 3 seeds
data1 = data.loc[(data["seed"] == seeds[0])]
data2 = data.loc[(data["seed"] == seeds[1])]
data3 = data.loc[(data["seed"] == seeds[2])]

#color for seed
#number x, p y
#use semilog

# plot the data
fig,ax = plt.subplots()
l1=ax.scatter(data1["N"], data1["p"], color='tab:blue', label = seeds[0])
l2=ax.scatter(data2["N"], data2["p"], color='tab:orange', label = seeds[1])
l3=ax.scatter(data3["N"], data3["p"], color='tab:red', label = seeds[2])
ax.set_xscale('log')

ax.set_xlabel('N (Trial Count)')
ax.set_ylabel('Probability')
ax.set_title('Part 3 Simulation Results')
ax.legend(handles=[l1,l2,l3], title = "Seeds")
#ax.set(xlim=(, 8), ylim=(0, 8))
plt.show()
