import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

df = pd.read_csv("exp_13_04_2023_10_07_00.csv")
df2 = pd.read_csv("exp_12_04_2023_13_41_52.csv")
df3 = pd.read_csv("exp_08_04_2023_03_07_11.csv")
df4 = pd.read_csv("exp_13_04_2023_20_10_54.csv")
#df5 = pd.read_csv("exp_17_04_2023_15_48_16.csv")
#df5 = pd.read_csv("exp_17_04_2023_18_30_01.csv")
df5 = pd.read_csv("exp_17_04_2023_21_32_01.csv")
df2["run"] = 1
df3["run"] = 2
df4["run"] = 3
df5["run"] = 4
print(len(df[df["num_success"] < 100]))
# df = df[df["num_success"] > 100]
# df2 = df2[df2["num_success"] > 100]
# df3 = df3[df3["num_success"] > 100]

df = pd.concat([df, df2], ignore_index=True, sort=False)
df = pd.concat([df, df3], ignore_index=True, sort=False)
df = pd.concat([df, df4], ignore_index=True, sort=False)
df = pd.concat([df, df5], ignore_index=True, sort=False)

#df = pd.read_csv("exp_04_04_2023_15_31_33.csv")
# limit to the first 100 entries per
def limit_rows(gr):
    return gr.head(100)

df2 = df2[df2["num_success"] > 100].groupby('num_rows').apply(limit_rows).reset_index(drop=True)
df3 = df3[df3["num_success"] > 100].groupby('num_rows').apply(limit_rows).reset_index(drop=True)
df4 = df4[df4["num_success"] > 100].groupby('num_rows').apply(limit_rows).reset_index(drop=True)
df5 = df5[df5["num_success"] > 100].groupby('num_rows').apply(limit_rows).reset_index(drop=True)
# remove degenerated runs ( <100 packets)
df = df[df["num_success"] > 100].groupby('num_rows').apply(limit_rows).reset_index(drop=True)

# remove Overhead > 20
df = df[df["num_rows"] < 2391]

df.to_csv("merged.csv", index=False)
df["num_rows"] -= (df["num_chunks"][0] - 1)  # -1 since we removed one packet each
ax = sns.violinplot(x='num_rows', y='num_success', data=df, cut=0, scale='count', showmedians=True)
df["num_rows"] += df["num_chunks"][0]
# add a horizontal line at the maximum value for each num_rows group
min_num_rows = df['num_rows'].min()
for i, (num_rows, group_data) in enumerate(df.groupby('num_rows')):
    max_value = max(group_data['num_rows'].max(), 0)
    group_violin = ax.collections[i]
    group_center = ax.get_xticks()[i]
    # group_width = group_violin.get_paths()[0].vertices[:, 0].max() - group_violin.get_paths()[0].vertices[:, 0].min()
    group_width = 1.0
    # group_center = group_violin.get_paths()[0].vertices[:, 0].mean()
    ax.hlines(max_value, group_center - group_width / 2, group_center + group_width / 2, linewidth=1, colors='red')

# add a custom legend
custom_legend = [Line2D([0], [0], color='red', lw=1, label='encoded packets')]
ax.legend(handles=custom_legend, bbox_to_anchor=(1, 0.1))

plt.grid(True)
# plt.title("Number of successful decodes for different number of encoded packet for a file encoded into 19 chunks")
# plt.title("for each run a different packet is removed from the equation system")

# plt.title("Number of non-critical packets")
plt.xlabel("Overhead")
# plt.ylabel("Number of successful decodes")
plt.ylabel("#non-critical packets")
plt.gcf().savefig('merged.pdf', bbox_inches='tight')
plt.gcf().savefig('merged.svg', format='svg', bbox_inches='tight')
plt.show()

# exp_04_04_2023_15_31_33.csv
