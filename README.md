# Seahorse

### Simple wrapper to produce graphs using python

Use both matplotlib, pandas, seaborn and custom graphical functions into the same library.
This lib was designed for my personnal use and does not intend to fullfill every need.

**Pro** : 
- Unified way to plot your data between mpl, pandas, seaborn and seahorse new functions
- Easy management of figures containing several axes
- Include new basic plot functions such as stacked barplot, circular barplot, non linear regressions ...
- Include new high level plot functions such as PyUpset or scatter distribution

### Installation

Using pip

```bash
pip install git+https://github.com/jsgounot/Seahorse.git
```

Or download / clone the github

```bash
git clone https://github.com/jsgounot/Seahorse.git
cd Seahorse

# Old python version
# python setup.py install --user

python -m pip install .
```

### Examples

**Custom container with unique functions** :

```python3
from seahorse import SContainer
from seahorse import sns

df = sns.load_dataset("tips")
sc = SContainer(df, 1, 3)

graph = sc.graph(0)
graph.sns.scatterplot(x="total_bill", y="tip", color="black")
graph.share_ax_lim()

sdf = df.groupby(['sex', 'smoker'])['total_bill'].mean().reset_index()
graph = sc.graph(1, sdf)
graph.sns.barplot(x="sex", y="total_bill", hue='smoker', linewidth=1, edgecolor='black')
graph.barplot_add_value(asint=True)
graph.change_bars_width(.35)
graph.ax.set_ylim((0, 25))
graph.remove_legend()

graph = sc.graph(2)
graph.sns.boxplot(x="day", y="tip", hue='smoker')
graph.change_boxplot_width(.9)
graph.add_xticks_ncount('day')
graph.make_annot(x="day", y="tip", hue="smoker", verbose=0, comparisons_correction="Bonferroni")
graph.legend_outside(title='Smoker?')

sc.set_size_inches(12, 4)
sc.tight_layout()
```

![Example image](https://github.com/jsgounot/Seahorse/blob/master/Examples/graphfun.png)

**Groupby Container** :

```python3
from seahorse import SContainer
from seahorse import sns

df = sns.load_dataset("tips")
sc = SContainer(df, 2, 2)

sc.groupby("day").sns.regplot("total_bill", "tip", color="black", scatter_kws={'s' : 8})
sc.set_labels("Total bill", "Tip")
sc.share_axes()

sc.set_size_inches(6, 6)
sc.tight_layout()
```

![Example image](https://github.com/jsgounot/Seahorse/blob/master/Examples/scontainer_gb.png)

**PyUpset** :

```python3
import pandas as pd
from seahorse import PyUpsetHue, sns

fname = "https://raw.githubusercontent.com/hms-dbmi/UpSetR/master/inst/extdata/movies.csv"
df = pd.read_csv(fname, sep=";", header=0)

# Data cleaning
df = df[[column for column in df.columns if column not in ["Watches", "ReleaseDate"]]]
df = pd.melt(df, id_vars=["Name", "AvgRating"], var_name="Kind", value_name="Found")
df = df[df["Found"] == 1].drop("Found", axis=1)
df["AvgRating"] = df["AvgRating"] // 1
df = df.sort_values('AvgRating')

def categorise_rates(rate) :
	if rate < 3 : return "low"
	if rate > 3 : return "high"
	return "medium"

# Apply rates for hue column
df["AvgRating"] = df["AvgRating"].apply(categorise_rates)

palette = sns.color_palette("Set2")

# Plot the data
graph = PyUpsetHue(df, key="Name", value="Kind", hue="AvgRating", 
	griddots=True, spacers=(.1, .25), palette=palette)

graph.set_size_inches(12, 5)
```

![Example image](https://github.com/jsgounot/Seahorse/blob/master/Examples/upset.png)