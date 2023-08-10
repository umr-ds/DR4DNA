import csv
import json

import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

font = {'size': 22}

# matplotlib.rc('font', **font)

df = pd.read_csv("output_sp_331.csv")

df["selected_rows"] = df["valid"] + df["invalid"]

# ax = sns.violinplot(x='avg_degree', y='num_success', data=df, cut=0, scale='count', showmedians=True)

df.groupby(["valid", "invalid"]).mean()
df.groupby(["invalid"]).mean()
df.groupby(["valid"]).mean()



df2 = pd.read_csv("output_sp_170.csv")

df2["selected_rows"] = df2["valid"] + df2["invalid"]

# ax = sns.violinplot(x='avg_degree', y='num_success', data=df2, cut=0, scale='count', showmedians=True)

df2.groupby(["valid", "invalid"]).mean()
df2.groupby(["invalid"]).mean()
df2.groupby(["valid"]).mean()

# TODO: change color of the lines to a gradient and use the gradient as the legend
color = cm.rainbow(np.linspace(0, 1, 20))
mymap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors', color)
min, max = (0, 19)
step = 1

# Setting up a colormap that's a simple transtion
# mymap = matplotlib.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0, 0], [0, 0]]
levels = range(min, max + step, step)
CS3 = plt.contourf(Z, levels, cmap=mymap)
plt.clf()

norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.rainbow)
cmap.set_array([])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32, 9), dpi=100)
# fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.grid(False)
# plt.ylabel("possible packets")
fig.supylabel('remaining packets')

for i, c in zip(range(20), color):
    tmp = df[df["invalid"] == i].groupby(["selected_rows"]).mean()
    p = ax[0].plot(tmp["valid"], tmp["avg_degree"], label=f"invalid={i}", c=cmap.to_rgba(i + 1))
# ax[0].set_aspect('equal')
# axins1 = inset_axes(ax, width='10%', height='2%', loc='lower right')
# fig.colorbar(CS3)
# cbar.ax.set_xticklabels(['0', '20'])
ax[0].grid(True)
# ax[0].ylabel("possible packets")
ax[0].set_xticks(np.arange(0, 19, step=2))
ax[0].set_xlabel("chunks tagged as valid")
# plt.colorbar

# plt.gcf().savefig('fixed_invalid_all_valid_vs_avg_degree331.pdf', bbox_inches='tight')
# plt.gcf().savefig('fixed_invalid_all_valid_vs_avg_degree331.svg', format='svg', bbox_inches='tight')
# plt.close()

for i, c in zip(range(20), color):
    tmp = df[df["valid"] == i].groupby(["selected_rows"]).mean()
    t_plt = ax[1].plot(tmp["invalid"], tmp["avg_degree"], label=f"valid={i}", c=cmap.to_rgba(i + 1))
# plt.legend()
# ax[1].set_aspect('equal')
ax[1].grid(True)
# fig.ylabel("possible packets")
ax[1].set_xticks(np.arange(0, 19, step=2))
ax[1].set_xlabel("chunks tagged as invalid")

# fig.subplots_adjust(right=0.8)
# fig.subplots_adjust(wspace=0.05)
plt.tight_layout()
fig.subplots_adjust(right=1)
# sub_ax = plt.axes([0.96, 0.55, 0.02, 0.3])
# plt.subplots_adjust(wspace=0.001, hspace=0.001)
cbar = fig.colorbar(cmap, ticks=np.arange(0, 20), ax=ax[:], pad=0.01)
cbar.set_label('number of opposite chunks tagged')
cbar.ax.invert_yaxis()
# fig.gcf().savefig('fixed_valid_all_valid_vs_avg_degree331.pdf', bbox_inches='tight')
# fig.gcf().savefig('fixed_valid_all_valid_vs_avg_degree331.svg', format='svg', bbox_inches='tight')

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

"""
for i, (num_rows, group_data) in enumerate(df.groupby('["valid", "invalid"]')):
    max_value = max(group_data['avg_degree'].max(), 0)
    group_violin = ax.collections[i]
    group_center = ax.get_xticks()[i]
    # group_width = group_violin.get_paths()[0].vertices[:, 0].max() - group_violin.get_paths()[0].vertices[:, 0].min()
    group_width = 1.0
    # group_center = group_violin.get_paths()[0].vertices[:, 0].mean()
    ax.hlines(max_value, group_center - group_width / 2, group_center + group_width / 2, linewidth=1, colors='red')

# add a custom legend
custom_legend = [Line2D([0], [0], color='red', lw=1, label='todo')]
ax.legend(handles=custom_legend, bbox_to_anchor=(1, 0.1))

plt.grid(True)
# plt.title("Number of successful decodes for different number of encoded packet for a file encoded into 19 chunks")
# plt.title("for each run a different packet is removed from the equation system")

# plt.title("Number of non-critical packets")
plt.xlabel("todo")
# plt.ylabel("Number of successful decodes")
plt.ylabel("todo")
plt.gcf().savefig('countrequired331.pdf', bbox_inches='tight')
plt.gcf().savefig('countrequired331.svg', format='svg', bbox_inches='tight')
plt.show()
"""
