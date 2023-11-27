import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm

font = {'size': 22}

"""
This script creates plots analyzing the effect of tagging packets as valid or invalid under various configurations.

It uses the output of the "CountRequiredTags"-plugin as the input. This plugin is disabled by default, so you have to
enable it by clicking on it in the plugin list.
"""

# matplotlib.rc('font', **font)

# Name of the file to analyze,
# "output_sp_331.csv" has been renamed from "output_all.csv" to prevent accidental overwriting it.
df = pd.read_csv("output_sp_331.csv")

df["selected_rows"] = df["valid"] + df["invalid"]

df.groupby(["valid", "invalid"]).mean()
df.groupby(["invalid"]).mean()
df.groupby(["valid"]).mean()

df2 = pd.read_csv("output_sp_170.csv")

df2["selected_rows"] = df2["valid"] + df2["invalid"]

df2.groupby(["valid", "invalid"]).mean()
df2.groupby(["invalid"]).mean()
df2.groupby(["valid"]).mean()

color = cm.rainbow(np.linspace(0, 1, 20))
mymap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors', color)
min, max = (0, 19)
step = 1

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0, 0], [0, 0]]
levels = range(min, max + step, step)
CS3 = plt.contourf(Z, levels, cmap=mymap)
plt.clf()

norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.rainbow)
cmap.set_array([])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32, 9), dpi=100)
fig.supylabel('remaining packets')

for i, c in zip(range(20), color):
    tmp = df[df["invalid"] == i].groupby(["selected_rows"]).mean()
    p = ax[0].plot(tmp["valid"], tmp["avg_degree"], label=f"invalid={i}", c=cmap.to_rgba(i + 1))
ax[0].grid(True)
ax[0].set_xticks(np.arange(0, 19, step=2))
ax[0].set_xlabel("chunks tagged as valid")

for i, c in zip(range(20), color):
    tmp = df[df["valid"] == i].groupby(["selected_rows"]).mean()
    t_plt = ax[1].plot(tmp["invalid"], tmp["avg_degree"], label=f"valid={i}", c=cmap.to_rgba(i + 1))

ax[1].grid(True)
ax[1].set_xticks(np.arange(0, 19, step=2))
ax[1].set_xlabel("chunks tagged as invalid")

plt.tight_layout()
fig.subplots_adjust(right=1)
cbar = fig.colorbar(cmap, ticks=np.arange(0, 20), ax=ax[:], pad=0.01)
cbar.set_label('number of opposite chunks tagged')
cbar.ax.invert_yaxis()

plt.gcf().savefig('fixed_combined_vs_avg_degree331.pdf', bbox_inches='tight')
plt.gcf().savefig('fixed_combined_vs_avg_degree331.svg', format='svg', bbox_inches='tight')
plt.close()

plt.plot(df.groupby(["valid"]).mean()["avg_degree"])
plt.xlabel("rows tagged as valid")
plt.ylabel("possible packets")
plt.grid(True)
plt.gcf().savefig('avg_degree_group_valid331.pdf', bbox_inches='tight')
plt.gcf().savefig('avg_degree_group_valid331.svg', format='svg', bbox_inches='tight')
plt.close()

plt.plot(df.groupby(["invalid"]).mean()["avg_degree"])
plt.xlabel("rows tagged as invalid")
plt.ylabel("possible packets")
plt.grid(True)
plt.gcf().savefig('avg_degree_group_invalid331.pdf', bbox_inches='tight')
plt.gcf().savefig('avg_degree_group_invalid331.svg', format='svg', bbox_inches='tight')
plt.close()

plt.plot(df2.groupby(["selected_rows"]).mean()["avg_degree"])
plt.plot(df.groupby(["selected_rows"]).mean()["avg_degree"])
plt.xlabel("tagged rows")
plt.ylabel("possible packets")
plt.grid(True)
plt.legend(["166 chunks", "331 chunks"])
plt.gcf().savefig('avg_degree_group_selected_rows_combined.pdf', bbox_inches='tight')
plt.gcf().savefig('avg_degree_group_selected_rows_combined.svg', format='svg', bbox_inches='tight')
plt.close()
