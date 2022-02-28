"""
Generate tornado diagrams

"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# input data

data_vals = np.array([[0.20, 0.80, 1.30],
                      [0.15, 0.80, 1.55],
                      [0.75, 0.80, 0.96],
                      [0.27, 0.80, 1.40]])

data_tags = np.array(['foo',
                      'bar',
                      'something',
                      'stuff'])

# calculate swing
swing = data_vals[:, 2] - data_vals[:, 0]

# determine ordering
idx = np.argsort(swing)

# generate plot
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.barh(data_tags[idx], data_vals[idx, 1] - data_vals[idx, 0], 0.15,
        left=data_vals[idx, 0], edgecolor='k',
        color='lightgray')
ax.barh(data_tags[idx], data_vals[idx, 2] - data_vals[idx, 1], 0.15,
        left=data_vals[idx, 1], edgecolor='k',
        color='lightgray')
plt.subplots_adjust(left=0.3)
ax.set(xlim=(0.0, 2.50))
ax.set(xlabel='Standard Deviation')
# plt.savefig('test.pdf')
plt.show()
