### Simple wrapper to produce graphs using python

Use both matplotlib, pandas, seaborn and custom graphical functions into the same library

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

![Example image](https://github.com/jsgounot/Seahorse/blob/master/example.png)

Under development