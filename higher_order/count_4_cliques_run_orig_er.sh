chmod +x count_4_cliques_collect_orig_er.sh
./count_4_cliques_collect_orig_er.sh > orig_er_4cliques_counts.txt

python3 - << 'PY'
import pandas as pd
df = pd.read_csv(
    "orig_er_4cliques_counts.txt",
    comment="#",
    sep=r"\s+",
    names=["dataset", "graph", "n_nodes", "n_4cliques"],
)
summary = (
    df.groupby("dataset")["n_4cliques"]
      .agg(["count", "mean", "std"])
      .reset_index()
)
print(summary.to_string(index=False))
PY