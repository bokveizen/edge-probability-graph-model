chmod +x count_4_cliques_collect.sh
./count_4_cliques_collect.sh > gen_4cliques_counts.txt

python3 - << 'PY'
import pandas as pd

df = pd.read_csv(
    "gen_4cliques_counts.txt",
    comment="#",
    sep=r"\s+",
    names=["model", "dataset", "graph", "n_nodes", "n_4cliques"],
)

summary = (
    df.groupby(["model", "dataset"])["n_4cliques"]
      .agg(["count", "mean", "std"])
      .reset_index()
)

print(summary.to_string(index=False))
PY