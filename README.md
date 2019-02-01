# Seahorse
  
### Simple wrapper to produce graphs using python

Use both matplotlib, pandas, seaborn and custom graphical functions into the same library.
This lib was designed for my personnal use and does not intend to fullfill every need.

**Pro** : 
- Unified way to plot your data between mpl, pandas, seaborn and seahorse new functions
- Easy management of figures containing several axes
- Include new basic plot functions such as stacked barplot, circular barplot, non linear regressions ...
- Include new high level plot functions such as PyUpset or scatter distribution

Example :

```python3

from seahorse import gwrap
from seahorse import SContainer

df = gwrap.sns.load_dataset("tips")
sc = SContainer(df, 1, 2)

sc.graph(0).sns.barplot("sex", "tip")
sc.graph(0).set_labels("Sex", "Tip")

sc.graph(1).shs.colored_regplot("total_bill", "tip", fit_reg=False, hue="sex")
sc.graph(1).set_labels("Total bill", "")
```

![Example image](https://github.com/jsgounot/Seahorse/blob/master/Examples/scontainer.png)

Another example :

```python3

import pandas as pd
from seahorse import PyUpsetHue

fname = "https://raw.githubusercontent.com/hms-dbmi/UpSetR/master/inst/extdata/movies.csv"
df = pd.read_csv(fname, sep=";", header=0)

# Data cleaning
df = df[[column for column in df.columns if column not in ["Watches", "ReleaseDate"]]]
df = pd.melt(df, id_vars=["Name", "AvgRating"], var_name="Kind", value_name="Found")
df = df[df["Found"] == 1].drop("Found", axis=1)
df["AvgRating"] = df["AvgRating"] // 1

def categorise_rates(rate) :
	if rate < 3 : return "low"
	if rate > 3 : return "high"
	return "medium"

# Apply rates for hue column
df["AvgRating"] = df["AvgRating"].apply(categorise_rates)

# Plot the data
graph = PyUpsetHue(df, key="Name", value="Kind", hue="AvgRating", griddots=True, spacers=(.1, .25))
graph.set_size(1200, 600)
```

![Example image](https://github.com/jsgounot/Seahorse/blob/master/Examples/upset.png)